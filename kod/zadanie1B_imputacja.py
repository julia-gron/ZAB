# =============================================================================
# Zaawansowana analityka biznesowa – metody imputacji danych
# Zadanie 1, wersja B (zmienna skokowa): czy kanał komunikacji A różnicuje szansę
# zakupu (Y_NEWDEPOSIT) w porównaniu z kanałem B?
#
# Program uruchamia się w całości:  python zadanie1B_imputacja.py
# Jedyne miejsce do zmiany przy innym położeniu danych: DATA_PATH zaraz poniżej.
# Wyniki (tabele .csv, wykresy .png) trafiają do folderu WYNIKI, a wszystko, co program
# wypisuje, jest dodatkowo zapisywane do pliku WYNIKI/wydruk_wynikow.txt.
#
# Sekcje programu:
#   0. Lokalizacja robocza, ziarno losowe, wczytanie i rekodowanie danych
#   1. Punkt 1 zadania: analiza struktury i braków danych
#   2. Punkt 2: zależność braku Y od cech, projekt modelu imputacyjnego
#   3. Punkt 3: model logitowy przy MCAR (kompletne przypadki) i MAR (imputacja wielokrotna)
#   4. Punkt 5: miara diagnostyczna imputacji i analiza wrażliwości (MNAR, IPW)
#   5. Kontrola krzyżowa tabel z sekcji 1
#   6. Punkt 4: podsumowanie wyników (tabela zbiorcza, wykres leśny)
# =============================================================================
# =============================================================================
# LOKALIZACJA ROBOCZA – jedyne miejsce do zmiany przy innym położeniu danych
# =============================================================================
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent   # folder projektu (nad folderem kod/)
DATA_PATH = BASE / "dane" / "kb05.csv"          # <- PEŁNA ścieżka do pliku z danymi
WYNIKI = BASE / "wyniki"                        # <- folder na wszystkie wyniki

import sys
import platform
import datetime
import warnings
import numpy as np
import pandas as pd
import scipy
import statsmodels
import matplotlib
matplotlib.use("Agg")                    # zapis wykresów do plików, bez okna
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from scipy.special import expit
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from patsy import dmatrix
from sklearn.metrics import roc_auc_score


# =============================================================================
# SEKCJA 0. KONFIGURACJA: lokalizacja robocza, ziarno, wczytanie i rekodowanie danych
# =============================================================================
# Polskie znaki i symbole (γ, ×, δ) na każdej konsoli, także w Windows
for strumien in (sys.stdout, sys.stderr):
    if hasattr(strumien, "reconfigure"):
        strumien.reconfigure(encoding="utf-8", errors="replace")

# 1. LOKALIZACJA ROBOCZA: zdefiniowana na początku programu (DATA_PATH, WYNIKI)
assert DATA_PATH.exists(), f"Nie znaleziono pliku z danymi: {DATA_PATH}"
WYNIKI.mkdir(parents=True, exist_ok=True)       # tworzony automatycznie

# 2. ZIARNO LOSOWE
SEED = 229090
np.random.seed(SEED)

# 3. ETYKIETY I POZIOMY REFERENCYJNE
# Pierwszy poziom na liście = poziom referencyjny w modelu.
CONTACT_LAB = {0: "0: brak komunikacji", 1: "A: kanał A", 2: "B: kanał B"}
CONTACT_LEVELS = ["2", "0", "1"]                              # ref = 2 (kanał B)
HL_LEVELS = ["yes", "no"]                                     # ref = yes
POUTCOME_LEVELS = ["unknown", "failure", "success", "other"]  # ref = unknown
EDU_LEVELS = ["secondary", "primary", "tertiary", "unknown"]  # ref = secondary
LEVELS = {"contact": CONTACT_LEVELS, "house_loan": HL_LEVELS,
          "poutcome": POUTCOME_LEVELS, "education": EDU_LEVELS}
KOLUMNY = ["y", "contact", "house_loan", "age", "education", "poutcome"]


# 4. WCZYTANIE DANYCH
def load_raw() -> pd.DataFrame:
    """Surowe dane jako tekst: nic nie jest po cichu zamieniane na brak."""
    # sep=None: separator (przecinek lub średnik) wykrywany automatycznie
    return pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False,
                       sep=None, engine="python")


def load_clean() -> pd.DataFrame:
    """Dane po rekodowaniu: kategorie z poziomami ref, y (NaN = brak), r."""
    d = load_raw()
    # Nazwy kolumn małymi literami (eksport z SAS bywa wielkimi): y_newdeposit -> y
    d.columns = d.columns.str.strip().str.lower()
    d = d.rename(columns={"y_newdeposit": "y"})
    brak_kolumn = set(KOLUMNY) - set(d.columns)
    assert not brak_kolumn, f"W pliku brakuje kolumn: {sorted(brak_kolumn)}"

    # Brak danych = puste pole lub kropka (zapis SAS), w KAŻDEJ kolumnie.
    # "unknown" NIE jest brakiem, to zwykła kategoria.
    d = d[KOLUMNY].apply(lambda s: s.str.strip()).replace({"": np.nan, ".": np.nan})

    # Zmienne liczbowe (tekst, którego nie da się zamienić na liczbę, zatrzyma program)
    d["y"] = pd.to_numeric(d["y"])
    d["age"] = pd.to_numeric(d["age"])
    assert d["y"].dropna().isin([0, 1]).all(), "Y musi mieć wartości 0 lub 1"

    # Cechy skokowe jako kategorie z ustalonymi poziomami (contact NIE jako liczba).
    # Puste pole zostaje brakiem, a nieznana etykieta (np. literówka) zatrzymuje program.
    for kol, poziomy in LEVELS.items():
        s = d[kol].str.lower() if kol != "contact" else d[kol]
        nieznane = set(s.dropna()) - set(poziomy)
        assert not nieznane, f"Nieznane kategorie w {kol}: {sorted(nieznane)}"
        d[kol] = pd.Categorical(s, categories=poziomy)
    assert d["contact"].notna().all(), "contact (grupa losowania) nie może mieć braków"

    d["r"] = d["y"].isna().astype(int)           # r = 1 oznacza brak Y
    return d[KOLUMNY + ["r"]]


# 5. WYDRUK WYNIKÓW: wszystko, co program wypisuje, trafia też do pliku
class Wydruk:
    """Zapisuje wypisywany tekst jednocześnie na ekran i do pliku z wydrukiem."""

    def __init__(self, plik):
        self.ekran, self.plik = sys.stdout, open(plik, "w", encoding="utf-8")

    def write(self, tekst):
        self.ekran.write(tekst)
        self.plik.write(tekst)

    def flush(self):
        self.ekran.flush()
        self.plik.flush()


sys.stdout = Wydruk(WYNIKI / "wydruk_wynikow.txt")
print("WYDRUK WYNIKÓW – Zadanie 1B, metody imputacji danych")
print(f"Data uruchomienia: {datetime.datetime.now():%Y-%m-%d %H:%M} | {platform.platform()}")
print("Wersje: Python", sys.version.split()[0], "| numpy", np.__version__, "| pandas",
      pd.__version__, "| scipy", scipy.__version__, "| statsmodels", statsmodels.__version__,
      "| matplotlib", matplotlib.__version__)

# 6. DANE CZYSTE
df = load_clean()
df.to_csv(WYNIKI / "dane_czyste.csv", index=False)
print("Dane czyste:", df.shape, "| braki Y:", int(df["r"].sum()))


print("\n" + "=" * 100 + "\nSEKCJA 1\n" + "=" * 100)
# =============================================================================
# SEKCJA 1. Punkt 1: analiza struktury i braków
# Lokalizacja danych i folderu wyników jest zdefiniowana w sekcji 0
# =============================================================================

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

print("Wersje: Python", sys.version.split()[0], "| pandas", pd.__version__,
      "| numpy", np.__version__, "| scipy", scipy.__version__,
      "| statsmodels", statsmodels.__version__, "| matplotlib", matplotlib.__version__)

# Wygląd wykresów (wspólny dla wszystkich)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})

# 0. PRZYGOTOWANIE
d = load_clean()
OUT = WYNIKI
LAB = CONTACT_LAB
d["kanal"] = d["contact"].astype(int).map(LAB)         # czytelne nazwy kanałów
ORDER = [LAB[0], LAB[1], LAB[2]]                        # stała kolejność kanałów
SHORT = ["0", "A", "B"]                                 # krótkie podpisy osi
COL = {LAB[0]: "#8c8c8c", LAB[1]: "#1f77b4", LAB[2]: "#ff7f0e"}
obs = d[d["r"] == 0]                                    # tylko obserwowane Y


def pl(x, n=1):
    """Liczba z przecinkiem dziesiętnym (zapis polski), np. 14,9."""
    return f"{x:.{n}f}".replace(".", ",")


def os_pl(ax, os="x"):
    """Przecinek dziesiętny na podziałce osi."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.xaxis if os == "x" else ax.yaxis).set_major_formatter(fmt)


def save(name, rect=(0, 0, 1, 1)):
    """Zapisuje bieżący wykres do folderu wyniki/ i zamyka go."""
    plt.tight_layout(rect=rect)
    plt.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close()


def prop_table(data, col, n_name, k_name):
    """Odsetek (średnia zmiennej 0/1) w kanałach + 95% przedział Wilsona."""
    g = data.groupby("kanal", observed=True)[col].agg(["count", "sum"])
    g = g.reindex(ORDER, fill_value=0)
    g["sum"] = g["sum"].astype(int)
    lo, hi = proportion_confint(g["sum"], g["count"], method="wilson")
    return pd.DataFrame({n_name: g["count"], k_name: g["sum"],
                         "odsetek": g["sum"] / g["count"],
                         "dolna granica 95%": lo, "górna granica 95%": hi})


def bar_ci(ax, tab, labels, capsize=5):
    """Słupki w % z asymetrycznymi słupkami błędów (przedział Wilsona), kolory kanałów."""
    est = 100 * tab["odsetek"].fillna(0)
    err_low = (100 * (tab["odsetek"] - tab["dolna granica 95%"])).fillna(0)
    err_high = (100 * (tab["górna granica 95%"] - tab["odsetek"])).fillna(0)
    return ax.bar(labels, est, yerr=[err_low, err_high], capsize=capsize,
                  color=[COL[k] for k in tab.index], error_kw={"elinewidth": 1.2})


# 1. KONTROLA DANYCH
print("\n### 1. Kontrola danych")
raw = load_raw()
for c in raw.columns:                                   # unikalne wartości
    u = sorted(raw[c].unique())
    opis = f"od {d.age.min():g} do {d.age.max():g}" if c.lower() == "age" else u
    print(f"{c}: {len(u)} unik. wartości, {opis}")
print("Wiek <18:", int((d.age < 18).sum()), "| >100:", int((d.age > 100).sum()))
print("Zduplikowane wiersze:", int(d.duplicated().sum()))
kol_id = [c for c in raw.columns if raw[c].nunique() == len(raw)]   # kolumna unikalna w każdym wierszu
if kol_id:
    print("Kolumna identyfikująca wiersze:", kol_id)
else:
    print("Kolumny w pliku:", list(raw.columns))
    print("Brak kolumny z identyfikatorem: identyczne wiersze to mogą być różne osoby o tych samych cechach, więc ich nie usuwamy.")
print("Liczność w grupach CONTACT:\n", d["kanal"].value_counts().reindex(ORDER))

# 2. CECHY CIĄGŁE (AGE): statystyki opisowe
print("\n### 2. Cecha ciągła AGE")


def desc(x):
    return pd.Series({"n": x.count(), "średnia": x.mean(), "odch.std": x.std(),
                      "min": x.min(), "Q1": x.quantile(.25), "mediana": x.median(),
                      "Q3": x.quantile(.75), "max": x.max()})


parts = {"Ogółem": desc(d.age)}
for k, g in d.groupby("kanal", observed=True):
    parts[f"CONTACT {k}"] = desc(g.age)
for k, g in d.groupby("r"):
    parts[f"r={k} ({'brak Y' if k else 'Y obserwowane'})"] = desc(g.age)
t_age = pd.DataFrame(parts).T.round(2)
print(t_age)
t_age.to_csv(OUT / "t1_age_opis.csv")

wiek = d.age.dropna()
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
# granice przedziałów liczone z danych (nic nie jest pomijane po cichu)
ax[0].hist(wiek, bins=np.arange(np.floor(wiek.min() / 3) * 3, wiek.max() + 3, 3),
           color="#4c72b0", edgecolor="white")
ax[0].axvline(wiek.mean(), color="k", ls="--", lw=1.2, label=f"średnia {pl(wiek.mean())} lat")
ax[0].axvline(wiek.median(), color="#d62728", ls=":", lw=1.8, label=f"mediana {pl(wiek.median(), 0)} lat")
ax[0].set(title="Rozkład wieku klientów", xlabel="Wiek (lata)", ylabel="Liczba klientów")
ax[0].legend(frameon=False)
ax[0].yaxis.grid(True, alpha=.3)
bp = ax[1].boxplot([d.age[d["kanal"] == k].dropna() for k in ORDER], showmeans=True,
                   patch_artist=True, medianprops={"color": "k"},
                   meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "k"})
for patch, k in zip(bp["boxes"], ORDER):
    patch.set_facecolor(COL[k])
    patch.set_alpha(.75)
ax[1].set_xticklabels(SHORT)
ax[1].set(title="Wiek według kanału komunikacji", ylabel="Wiek (lata)",
          xlabel="CONTACT (kanał)\n◇ średnia, linia w pudełku: mediana")
ax[1].yaxis.grid(True, alpha=.3)
save("w1_age_hist_box.png")

# 3. CECHY SKOKOWE 
print("\n### 3. Cechy skokowe")
cechy = {c: d[c].astype(object).fillna("brak (.)")
         for c in ["contact", "house_loan", "education", "poutcome"]}
cechy["y"] = d.y.map({0: "0", 1: "1"}).fillna("brak (.)")
rows = []
for nazwa, s in cechy.items():
    for kategoria, n in s.value_counts().items():
        rows.append([nazwa, kategoria, n, 100 * n / len(d)])
t_cat = pd.DataFrame(rows, columns=["zmienna", "kategoria", "n", "%"]).round(2)
print(t_cat.to_string(index=False))
t_cat.to_csv(OUT / "t2_cechy_skokowe.csv", index=False)

# wykres liczności cech skokowych (4 panele)
tyt = {"contact": "Kanał komunikacji", "house_loan": "Kredyt mieszkaniowy",
       "education": "Wykształcenie", "poutcome": "Wynik poprzedniej kampanii"}
fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, (nazwa, s) in zip(axs, list(cechy.items())[:4]):
    vc = s.value_counts().sort_values()
    nazwy = [LAB[int(i)] if nazwa == "contact" else str(i) for i in vc.index]   # kanały z opisem
    ax.barh(nazwy, 100 * vc / len(d), color="#4c72b0")
    for i, v in enumerate(vc):
        ax.text(100 * v / len(d) + 1, i, f"{pl(100 * v / len(d))}% (n={v})", va="center", fontsize=8)
    ax.set(title=tyt[nazwa], xlabel="% klientów", xlim=(0, 118))
    ax.xaxis.grid(True, alpha=.3)
    os_pl(ax)
save("w7_cechy_skokowe.png")

for c in ["house_loan", "education", "poutcome"]:        # tabele krzyżowe z CONTACT
    ct = pd.crosstab(d[c], d["kanal"])[ORDER]            # (puste pola pomijane)
    pct = (100 * ct / ct.sum()).round(1)                 # % w kanale
    chi = stats.chi2_contingency(ct)
    print(f"\n{c} x CONTACT (% w kanale), chi2={chi[0]:.2f}, p={chi[1]:.3f}\n{pct}")
    pd.concat({"n": ct, "% w kanale": pct}, axis=1).to_csv(OUT / f"t3_krzyzowa_{c}.csv")

# 4. WYNIK BADANIA: % Y=1 według CONTACT (obserwowane Y)
print("\n### 4. Wynik badania")
t_y = prop_table(obs, "y", "n_obs", "Y=1")
print(t_y.round(4))
t_y.to_csv(OUT / "t4_y_wg_contact.csv")

fig, ax = plt.subplots(figsize=(6.8, 4.4))
bar_ci(ax, t_y, ORDER)
for i, (n, p, hi) in enumerate(zip(t_y["n_obs"], t_y["odsetek"], t_y["górna granica 95%"])):
    ax.text(i, 100 * hi + 1, f"{pl(100 * p)}%\n(n={int(n)})", ha="center", va="bottom", fontsize=9)
ax.set(title="Odsetek zakupów (Y=1) według kanału", ylabel="% klientów z zakupem",
       ylim=(0, 100 * t_y["górna granica 95%"].max() + 6))
ax.yaxis.grid(True, alpha=.3)
os_pl(ax, "y")
ax.text(0, -.17, "Obserwowane Y; słupki błędów: 95% przedział ufności Wilsona.",
        transform=ax.transAxes, fontsize=8, color="0.3")
save("w2_y_wg_contact.png")


def y_w_podgrupach(var, tytul, fname):
    """Ten sam wykres co wyżej, osobno dla każdej kategorii cechy `var`."""
    kategorie = list(d[var].cat.categories)
    tabs = {lv: prop_table(obs[obs[var] == lv], "y", "n_obs", "Y=1") for lv in kategorie}
    ymax = 100 * max(t["górna granica 95%"].max() for t in tabs.values())   # wspólna skala osi Y
    fig, axs = plt.subplots(1, len(kategorie), figsize=(3.4 * len(kategorie), 4.4), sharey=True)
    tabele = []
    for ax, (lv, tab) in zip(np.atleast_1d(axs), tabs.items()):
        bar_ci(ax, tab, SHORT, capsize=4)
        for i, (n, hi) in enumerate(zip(tab["n_obs"], tab["górna granica 95%"].fillna(0))):
            ax.text(i, 100 * hi + .03 * ymax, f"n={int(n)}", ha="center", fontsize=8)
        ax.set_ylim(0, ymax * 1.18)                      # zapas na podpisy n=...
        ax.set_title(f"{lv}", pad=10)
        ax.set_xlabel("CONTACT (kanał)")
        ax.yaxis.grid(True, alpha=.3)
        os_pl(ax, "y")
        tab["podgrupa"] = f"{var}={lv}"
        tabele.append(tab.reset_index())
    np.atleast_1d(axs)[0].set_ylabel("% klientów z zakupem (Y=1)")
    fig.suptitle(f"Odsetek zakupów według kanału i kategorii: {tytul}", fontweight="bold")
    fig.text(.01, .01, "Obserwowane Y; słupki błędów: 95% przedział ufności Wilsona.",
             fontsize=8, color="0.3")
    save(fname, rect=(0, .05, 1, .94))
    return pd.concat(tabele)


t_hl = y_w_podgrupach("house_loan", "kredyt mieszkaniowy", "w3_y_contact_houseloan.png")
t_po = y_w_podgrupach("poutcome", "wynik poprzedniej kampanii", "w4_y_contact_poutcome.png")
print("\nPodgrupy house_loan:\n", t_hl.round(3).to_string(index=False))
print("\nPodgrupy poutcome:\n", t_po.round(3).to_string(index=False))
pd.concat([t_hl, t_po]).to_csv(OUT / "t5_y_podgrupy.csv", index=False)

# 5. BRAKI DANYCH
print("\n### 5. Braki danych")
cols = ["y", "contact", "house_loan", "age", "education", "poutcome"]
t_braki = pd.DataFrame({"braki": d[cols].isna().sum()})
t_braki["% braków"] = (100 * t_braki["braki"] / len(d)).round(2)
t_braki["'unknown'"] = [int((d[c] == "unknown").sum()) if c in ("education", "poutcome") else "-"
                        for c in t_braki.index]          # "-" = zmienna bez kategorii unknown
print(t_braki)
t_braki.to_csv(OUT / "t6_braki_zmienne.csv")
print("Braki tylko w Y:", bool(d[[c for c in cols if c != "y"]].isna().sum().sum() == 0))

# Wzorce braków (1 = brak). Brakuje tylko y => 2 wzorce, monotoniczność trywialna.
pat = d[cols].isna().astype(int).value_counts().rename("n").reset_index()
pat["% próby"] = (100 * pat["n"] / len(d)).round(2)
print("Wzorce braków (1 = brak):\n", pat)
pat.to_csv(OUT / "t7_wzorce_brakow.csv", index=False)

t_m = prop_table(d, "r", "n", "braki Y")                 # braki Y wg kanału
chi_m = stats.chi2_contingency(pd.crosstab(d["kanal"], d["r"]))
print(f"Braki Y wg CONTACT:\n{t_m.round(4)}\nchi2={chi_m[0]:.2f}, p={chi_m[1]:.3f}")
t_m.to_csv(OUT / "t8_braki_wg_contact.csv")

fig, ax = plt.subplots(figsize=(6.8, 4.4))
bar_ci(ax, t_m, ORDER)
sr = 100 * d["r"].mean()
ax.axhline(sr, ls="--", c="k", lw=1)
for i, (n, p, hi) in enumerate(zip(t_m["n"], t_m["odsetek"], t_m["górna granica 95%"])):
    ax.text(i, 100 * hi + 1, f"{pl(100 * p)}%\n(n={int(n)})", ha="center", va="bottom", fontsize=9)
ax.set(title="Odsetek braków Y według kanału", ylabel="% klientów bez wartości Y",
       ylim=(0, 100 * t_m["górna granica 95%"].max() + 8))
ax.yaxis.grid(True, alpha=.3)
os_pl(ax, "y")
ax.text(0, -.21, f"Linia przerywana: średni odsetek braków w całej próbie ({pl(sr)}%). "
        f"Słupki błędów: 95% przedział ufności Wilsona.\nTest chi-kwadrat różnic między kanałami: p = {pl(chi_m[1], 2)}.",
        transform=ax.transAxes, fontsize=8, color="0.3", va="top")
save("w5_braki_wg_contact.png")

# 6. RÓWNOWAGA LOSOWANIA: SMD między grupami CONTACT
print("\n### 6. SMD (próg 0,1)")
# Cechy skokowe -> zmienne 0/1, bez poziomów referencyjnych
dummies = pd.get_dummies(d[["house_loan", "poutcome", "education"]], dtype=float)
X = pd.concat([d[["age"]],
               dummies.drop(columns=["house_loan_yes", "poutcome_unknown",
                                     "education_secondary"])], axis=1)


def smd(a, b):
    """|różnica średnich| / sqrt((wariancja_a + wariancja_b) / 2), dla każdej kolumny."""
    return (a.mean() - b.mean()).abs() / np.sqrt((a.var() + b.var()) / 2)


pary = {"A vs B": ("1", "2"), "A vs 0": ("1", "0"), "B vs 0": ("2", "0")}
proby = {"pełna próba": pd.Series(True, index=d.index), "obserwowane Y": d["r"] == 0}
t_smd = pd.DataFrame({
    (proba, para): smd(X[w & (d.contact == g1)], X[w & (d.contact == g2)])
    for proba, w in proby.items() for para, (g1, g2) in pary.items()})
print(t_smd.round(3))
t_smd.round(4).to_csv(OUT / "t9_smd.csv")
print("Maks. SMD pełna próba:", t_smd["pełna próba"].max().round(3).to_dict(),
      "\nMaks. SMD obserwowane:", t_smd["obserwowane Y"].max().round(3).to_dict())

etyk = {"age": "wiek", "house_loan_no": "brak kredytu mieszkaniowego",
        "poutcome_failure": "poprzednia kampania: porażka",
        "poutcome_success": "poprzednia kampania: sukces",
        "poutcome_other": "poprzednia kampania: inny wynik",
        "education_primary": "wykształcenie podstawowe",
        "education_tertiary": "wykształcenie wyższe",
        "education_unknown": "wykształcenie nieznane"}
poz = np.arange(len(X.columns))[::-1]                    # pierwsza cecha na górze
xmax = max(.15, np.nanmax(t_smd.values) * 1.12)
fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.8), sharey=True, sharex=True)
for ax, para in zip(axs, pary):
    a, b = t_smd[("pełna próba", para)], t_smd[("obserwowane Y", para)]
    ax.hlines(poz, np.minimum(a, b), np.maximum(a, b), color="0.75", lw=2, zorder=1)
    ax.scatter(a, poz, s=45, color="#1f77b4", zorder=2, label="pełna próba")
    ax.scatter(b, poz, s=45, marker="D", facecolor="white", edgecolor="#d62728",
               linewidth=1.5, zorder=3, label="tylko obserwowane Y")
    ax.axvline(.1, ls="--", c="k", lw=1, label="próg 0,1" if ax is axs[0] else None)
    ax.set_title(para.replace("vs", "wobec"))
    ax.set_xlabel("|SMD|")
    ax.set_xlim(-.02, xmax)                              # zapas, żeby punkty w 0 nie były ucięte
    ax.xaxis.grid(True, alpha=.3)
    os_pl(ax)
axs[0].set_yticks(poz)
axs[0].set_yticklabels([etyk.get(c, c) for c in X.columns])
axs[0].legend(frameon=False, loc="lower right", fontsize=8)
fig.suptitle("Standaryzowane różnice średnich (SMD) między kanałami", fontweight="bold")
fig.text(.01, .01, "Poziomy odniesienia: kredyt mieszkaniowy tak, poprzednia kampania "
         "nieznana, wykształcenie średnie. SMD > 0,1 oznacza niezrównoważone grupy.",
         fontsize=8, color="0.3")
save("w6_smd.png", rect=(0, .04, 1, .94))
print("\nPliki zapisane w:", OUT)


print("\n" + "=" * 100 + "\nSEKCJA 2\n" + "=" * 100)
# =============================================================================
# SEKCJA 2. Punkt 2: mechanizm braków danych w Y
# Czy prawdopodobieństwo braku Y zależy od pozostałych cech i czy są cechy,
# za pomocą których imputacja może skorygować oszacowania?
# Zastosowane metody:
#   Metoda I  - test t różnicy średnich wieku między grupami braku (PROC TTEST)
#   Metoda II - test chi-kwadrat i V Cramera dla cech skokowych (PROC FREQ / CHISQ)
#   Model logistyczny prawdopodobieństwa obserwacji Y, P(M = 0) (PROC LOGISTIC)
# Lokalizacja danych i folderu wyników jest zdefiniowana w sekcji 0.
# Pliki wynikowe tej części mają przedrostek s2_.
# =============================================================================

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

print("Wersje: Python", sys.version.split()[0], "| pandas", pd.__version__,
      "| numpy", np.__version__, "| scipy", scipy.__version__,
      "| statsmodels", statsmodels.__version__, "| matplotlib", matplotlib.__version__)

# Wygląd wykresów taki sam jak w sekcji 1
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})

# 0. PRZYGOTOWANIE
d = load_clean()
OUT = WYNIKI
LAB = CONTACT_LAB
d["kanal"] = d["contact"].astype(int).map(LAB)
ORDER = [LAB[0], LAB[1], LAB[2]]
COL = {LAB[0]: "#8c8c8c", LAB[1]: "#1f77b4", LAB[2]: "#ff7f0e"}   # kolory kanałów jak u Osoby 1
NIEBIESKI = "#4c72b0"                                              # kolor słupków jak u Osoby 1
# Wskaźnik braku M (1 = brak Y). W kodzie nazywa się r (sekcja 0).
# Model logistyczny opisuje prawdopodobieństwo obserwacji Y: P(M = 0).
d["obs_y"] = 1 - d["r"]                                   # 1 = Y zaobserwowane
# Wiek w przedziałach (do tabel i wykresów)
PRZ, ETP = [17, 30, 40, 50, 60, 100], ["18-30", "31-40", "41-50", "51-60", "61+"]
d["wiek_p"] = pd.cut(d["age"], PRZ, labels=ETP)
obs = d[d["r"] == 0]                                      # tylko obserwowane Y
SR = d["r"].mean()                                        # średni odsetek braków
ALFA = 0.05
print(f"n = {len(d)}, braki Y: {int(d.r.sum())} ({100 * SR:.2f}%)")


def pl(x, n=1):
    """Liczba z przecinkiem dziesiętnym (zapis polski), np. 25,5."""
    return f"{x:.{n}f}".replace(".", ",")


def os_pl(ax, os="y"):
    """Przecinek dziesiętny na podziałce osi."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.xaxis if os == "x" else ax.yaxis).set_major_formatter(fmt)


def save(name, rect=(0, 0, 1, 1)):
    """Zapisuje bieżący wykres do folderu wyniki/ i zamyka go."""
    plt.tight_layout(rect=rect)
    plt.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close()


def odsetki(data, kol, grupa):
    """Odsetek (średnia zmiennej 0/1) w kategoriach `grupa` + 95% przedział Wilsona."""
    g = data.groupby(grupa, observed=True)[kol].agg(["count", "sum"])
    lo, hi = proportion_confint(g["sum"], g["count"], method="wilson")
    return pd.DataFrame({"n": g["count"], "liczba": g["sum"].astype(int),
                         "odsetek": g["sum"] / g["count"], "dolna 95%": lo, "górna 95%": hi})


def logit(formula, data):
    """Regresja logistyczna (odpowiednik PROC LOGISTIC)."""
    m = smf.logit(formula, data=data).fit(disp=0, maxiter=100)
    assert m.mle_retvals["converged"], f"Brak zbieżności: {formula}"
    return m


def type3(m):
    """Testy Walda dla całych cech (jak tabela 'Type 3 Analysis of Effects' w PROC LOGISTIC)."""
    t = m.wald_test_terms(scalar=True).summary_frame()
    t = t.rename(columns={"chi2": "chi2 Walda", "P>chi2": "p", "df constraint": "df"})
    return t.drop(index="Intercept")[["chi2 Walda", "df", "p"]]


def ilorazy_szans(m):
    """Ilorazy szans z 95% przedziałami ufności (jak 'Odds Ratio Estimates' w PROC LOGISTIC)."""
    ci = m.conf_int()
    t = pd.DataFrame({"OR": np.exp(m.params), "dolna 95%": np.exp(ci[0]),
                      "górna 95%": np.exp(ci[1]), "p": m.pvalues})
    return t.drop(index="Intercept")


def ogolny_test(m, data, y):
    """Test ilorazu wiarygodności H0: wszystkie parametry = 0 ('Testing Global Null
    Hypothesis' w PROC LOGISTIC) i statystyka c (pole pod krzywą ROC)."""
    c = roc_auc_score(data[y], m.predict(data))
    return {"LR chi2": m.llr, "df": int(m.df_model), "p": m.llr_pvalue, "c (AUC)": c}


# =============================================================================
# 1. METODA I: TEST t RÓŻNICY ŚREDNICH WIEKU (PROC TTEST)
# =============================================================================
print("\n### 1. Metoda I: test t dla wieku według r")
a0, a1 = d.loc[d.r == 0, "age"], d.loc[d.r == 1, "age"]
tt = stats.ttest_ind(a0, a1)                              # test t Studenta, dwustronny
t_ttest = pd.DataFrame({
    "grupa": ["Y obserwowane (r = 0)", "brak Y (r = 1)"],
    "n": [len(a0), len(a1)], "średnia wieku": [a0.mean(), a1.mean()],
    "odch. std.": [a0.std(), a1.std()]})
print(t_ttest.round(2).to_string(index=False))
print(f"Różnica średnich = {a0.mean() - a1.mean():.2f} roku, t = {tt.statistic:.3f}, "
      f"df = {len(a0) + len(a1) - 2}, p = {tt.pvalue:.4f}")
t_ttest.round(4).to_csv(OUT / "s2_t1_wiek_wg_r.csv", index=False)

# Uzupełnienie testu t: test Manna-Whitneya (czy rozkład wieku jest przesunięty)
# i test Kołmogorowa-Smirnowa (czy rozkłady wieku mają ten sam kształt)
mw = stats.mannwhitneyu(a0, a1)
ks = stats.ks_2samp(a0, a1)
t_testy_wiek = pd.DataFrame({
    "test": ["t Studenta (równość średnich)", "Manna-Whitneya (przesunięcie rozkładu)",
             "Kołmogorowa-Smirnowa (zgodność rozkładów)"],
    "statystyka": [tt.statistic, mw.statistic, ks.statistic], "p": [tt.pvalue, mw.pvalue, ks.pvalue]})
print(t_testy_wiek.round(4).to_string(index=False))
t_testy_wiek.round(4).to_csv(OUT / "s2_t1_testy_wiek.csv", index=False)

# =============================================================================
# 2. METODA II: TEST CHI-KWADRAT DLA CECH SKOKOWYCH (PROC FREQ / CHISQ)
# =============================================================================
print("\n### 2. Metoda II: odsetek braków Y w kategoriach cech, test chi-kwadrat")
CECHY_T = {"kanal": "CONTACT", "house_loan": "HOUSING_LOAN", "education": "EDUCATION",
           "poutcome": "POUTCOME", "wiek_p": "AGE (przedziały)"}
rows, testy = [], []
for kol, nazwa in CECHY_T.items():
    ct = pd.crosstab(d[kol], d["r"])
    chi2, p, df, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))   # V Cramera
    testy.append([nazwa, chi2, df, p, v])
    t = odsetki(d, "r", kol)
    for kat, w in t.iterrows():
        rows.append([nazwa, str(kat), w["n"], w["liczba"], w["odsetek"], w["dolna 95%"], w["górna 95%"]])
t_kat = pd.DataFrame(rows, columns=["cecha", "kategoria", "n", "braki Y", "odsetek braków",
                                    "dolna 95%", "górna 95%"])
t_chi = pd.DataFrame(testy, columns=["cecha", "chi2", "df", "p", "V Cramera"]).set_index("cecha")
print(t_kat.round(4).to_string(index=False))
print("\nTesty chi-kwadrat niezależności r od cechy:\n", t_chi.round(4))
t_kat.round(4).to_csv(OUT / "s2_t2_braki_wg_kategorii.csv", index=False)
t_chi.round(4).to_csv(OUT / "s2_t3_chi2_braki.csv")

# Wykres: odsetek braków Y w kategoriach każdej cechy (styl jak rysunek 3 Osoby 1)
tyt = {"CONTACT": "Kanał komunikacji", "HOUSING_LOAN": "Kredyt mieszkaniowy",
       "EDUCATION": "Wykształcenie", "POUTCOME": "Wynik poprzedniej kampanii",
       "AGE (przedziały)": "Wiek (lata)"}
krotko = {LAB[0]: "0", LAB[1]: "A", LAB[2]: "B"}
fig, axs = plt.subplots(1, 5, figsize=(17, 4.6), sharey=True,
                        gridspec_kw={"width_ratios": [3, 2, 4, 4, 5]})
for ax, nazwa in zip(axs, CECHY_T.values()):
    t = t_kat[t_kat.cecha == nazwa]
    lab = [krotko.get(k, k) for k in t.kategoria]
    kolory = [COL[k] for k in t.kategoria] if nazwa == "CONTACT" else NIEBIESKI
    est = 100 * t["odsetek braków"]
    err = [100 * (t["odsetek braków"] - t["dolna 95%"]), 100 * (t["górna 95%"] - t["odsetek braków"])]
    ax.bar(lab, est, yerr=err, capsize=4, color=kolory, error_kw={"elinewidth": 1.2})
    ax.axhline(100 * SR, ls="--", c="k", lw=1)
    for i, (e, hi) in enumerate(zip(est, 100 * t["górna 95%"])):
        ax.text(i, hi + .8, f"{pl(e)}%", ha="center", fontsize=8)
    ax.set_title(f"{tyt[nazwa]}\n(chi-kwadrat: p = {pl(t_chi.loc[nazwa, 'p'], 2)})")
    ax.set_ylim(0, 42)
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.grid(True, alpha=.3)
    os_pl(ax)
axs[0].set_ylabel("% klientów bez wartości Y")
fig.suptitle("Odsetek braków Y w kategoriach cech", fontweight="bold")
fig.text(.01, .01, f"Linia przerywana: średni odsetek braków w całej próbie ({pl(100 * SR)}%). "
         "Słupki błędów: 95% przedział ufności Wilsona.", fontsize=8, color="0.3")
save("s2_w1_braki_wg_cech.png", rect=(0, .04, 1, .95))

# =============================================================================
# 3. MODEL LOGISTYCZNY PRAWDOPODOBIEŃSTWA OBSERWACJI Y, P(M = 0)
# =============================================================================
print("\n### 3. Model logistyczny P(Y zaobserwowane | cechy)")
GLOWNE = "contact + age + house_loan + poutcome + education"
m_r = logit(f"obs_y ~ {GLOWNE}", d)
og_r = ogolny_test(m_r, d, "obs_y")
t3_r = type3(m_r)
or_r = ilorazy_szans(m_r)
print(f"Test globalny LR: chi2 = {og_r['LR chi2']:.2f}, df = {og_r['df']}, p = {og_r['p']:.4f}; "
      f"c = {og_r['c (AUC)']:.3f}")
print("Testy Walda (Type 3):\n", t3_r.round(4))
print("Ilorazy szans obserwacji Y:\n", or_r.round(4))
t3_r.round(4).to_csv(OUT / "s2_t4_model_braku_type3.csv")
or_r.round(4).to_csv(OUT / "s2_t5_model_braku_or.csv")

# 3a. Czy cechy działają na brak Y tak samo w każdym kanale? Interakcje z CONTACT.
# Interakcji CONTACT x POUTCOME nie badamy: w kanale 0 aż 99,3% klientów ma poutcome = unknown.
INTER = "contact:age + contact:house_loan + contact:education"
m_r_int = logit(f"obs_y ~ {GLOWNE} + {INTER}", d)
t3_r_int = type3(m_r_int).loc[["contact:age", "contact:house_loan", "contact:education"]]
print("\nInterakcje w modelu braku (Wald, Type 3):\n", t3_r_int.round(4))

# Odsetek braków według wieku i kanału (opis interakcji CONTACT x AGE)
t_wk = odsetki(d, "r", ["kanal", "wiek_p"]).reset_index()
print("\nOdsetek braków Y według kanału i wieku:\n", t_wk.round(3).to_string(index=False))
t_wk.round(4).to_csv(OUT / "s2_t6_braki_wiek_kanal.csv", index=False)

# 3b. P(Y zaobserwowane | X) dla każdej obserwacji -> wagi IPW (Osoba 4)
p_obs = pd.DataFrame({"wiersz": np.arange(len(d)), "r": d["r"], "p_obs": m_r.predict(d)})
p_obs.to_csv(OUT / "s2_p_obserwacji.csv", index=False)
print(f"\nP(r = 0 | X) zapisane do s2_p_obserwacji.csv, zakres {p_obs.p_obs.min():.3f}-"
      f"{p_obs.p_obs.max():.3f}")

# =============================================================================
# 4. ZWIĄZEK CECH Z Y (tylko obserwowane Y)
# =============================================================================
print("\n### 4. Model logistyczny Y na obserwowanych Y, ze wszystkimi cechami")
m_y = logit(f"y ~ {GLOWNE}", obs)
og_y = ogolny_test(m_y, obs, "y")
t3_y = type3(m_y)
or_y = ilorazy_szans(m_y)
print(f"Test globalny LR: chi2 = {og_y['LR chi2']:.2f}, df = {og_y['df']}, p = {og_y['p']:.4g}; "
      f"c = {og_y['c (AUC)']:.3f}")
print("Testy Walda (Type 3):\n", t3_y.round(4))
print("Ilorazy szans zakupu:\n", or_y.round(4))
t3_y.round(4).to_csv(OUT / "s2_t7_model_y_type3.csv")
or_y.round(4).to_csv(OUT / "s2_t8_model_y_or.csv")

m_y_int = logit(f"y ~ {GLOWNE} + {INTER}", obs)
t3_y_int = type3(m_y_int).loc[["contact:age", "contact:house_loan", "contact:education"]]
print("\nInterakcje w modelu Y (Wald, Type 3):\n", t3_y_int.round(4))

t_zk = odsetki(obs, "y", ["kanal", "wiek_p"]).reset_index()
print("\nOdsetek zakupów (obserwowane Y) według kanału i wieku:\n", t_zk.round(3).to_string(index=False))
t_zk.round(4).to_csv(OUT / "s2_t9_zakup_wiek_kanal.csv", index=False)

# Zbiorcza tabela testów ogólnych obu modeli
pd.DataFrame({"model braku (P obserwacji Y)": og_r, "model Y (obserwowane Y)": og_y}).T.round(4) \
    .to_csv(OUT / "s2_t10_testy_globalne.csv")

# Porównanie modeli bez interakcji i z interakcjami: -2 log L, AIC (jak 'Model Fit
# Statistics' w PROC LOGISTIC) oraz test ilorazu wiarygodności dla modeli zagnieżdżonych
print("\n### 4a. Porównanie modeli bez interakcji i z interakcjami")
rows = []
for nazwa, m0, m1 in [("model braku (P obserwacji Y)", m_r, m_r_int),
                      ("model Y (obserwowane Y)", m_y, m_y_int)]:
    lr = 2 * (m1.llf - m0.llf)
    df_lr = int(m1.df_model - m0.df_model)
    rows.append([nazwa, -2 * m0.llf, m0.aic, -2 * m1.llf, m1.aic, lr, df_lr, stats.chi2.sf(lr, df_lr)])
t_por = pd.DataFrame(rows, columns=["model", "-2 log L bez interakcji", "AIC bez interakcji",
                                    "-2 log L z interakcjami", "AIC z interakcjami",
                                    "LR chi2", "df", "p"]).set_index("model")
print(t_por.round(4))
t_por.round(4).to_csv(OUT / "s2_t10a_porownanie_modeli.csv")


# Wykresy: odsetek braków i odsetek zakupów według wieku i kanału (słupki w kolorach kanałów)
def slupki_wiek_kanal(t, tytul, oslabel, plik, linia=None):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    szer = .27
    male = t[t["n"] < 20]                                # grupy pomijane na wykresie
    opis_male = "; ".join(f"{k.split(':')[0]}, wiek {w}: n = {n}"
                          for k, w, n in zip(male.kanal, male.wiek_p, male.n))
    for j, k in enumerate(ORDER):
        tk = t[t.kanal == k].set_index("wiek_p").reindex(ETP)
        tk = tk.where(tk["n"] >= 20)                     # przedziały z n < 20 pomijamy
        x = np.arange(len(ETP)) + (j - 1) * szer
        est = 100 * tk["odsetek"]
        err = [100 * (tk["odsetek"] - tk["dolna 95%"]), 100 * (tk["górna 95%"] - tk["odsetek"])]
        ax.bar(x, est, szer, yerr=err, capsize=3, color=COL[k], label=k,
               error_kw={"elinewidth": 1})
    if linia is not None:
        ax.axhline(100 * linia, ls="--", c="k", lw=1)
    ax.set_xticks(np.arange(len(ETP)))
    ax.set_xticklabels(ETP)
    ax.set(title=tytul, xlabel="Wiek (lata)", ylabel=oslabel)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.yaxis.grid(True, alpha=.3)
    os_pl(ax)
    przypis = f"Słupki błędów: 95% przedział ufności Wilsona. Pominięto grupy z n < 20 (kanał {opis_male})."
    if linia is not None:
        przypis = f"Linia przerywana: średni odsetek braków ({pl(100 * linia)}%). " + przypis
    fig.text(.01, .01, przypis, fontsize=8, color="0.3")
    save(plik, rect=(0, .05, 1, 1))


slupki_wiek_kanal(t_wk, "Odsetek braków Y według wieku i kanału", "% klientów bez wartości Y",
                  "s2_w2_braki_wiek_kanal.png", linia=SR)
slupki_wiek_kanal(t_zk, "Odsetek zakupów (Y=1) według wieku i kanału", "% klientów z zakupem",
                  "s2_w3_zakup_wiek_kanal.png")

# =============================================================================
# 5. TABELA DECYZYJNA I PROJEKT MODELU IMPUTACYJNEGO
# =============================================================================
print("\n### 5. Tabela decyzyjna")


def rola(z_r, z_y):
    """Związek z brakiem i z Y -> usuwa obciążenie; tylko z Y -> poprawia precyzję;
    tylko z brakiem -> niewiele daje; bez związku -> bez wpływu."""
    if z_r and z_y:
        return "usuwa obciążenie"
    if z_y:
        return "poprawia precyzję"
    if z_r:
        return "niewiele daje"
    return "bez wpływu"


W_DOC = {"contact", "age", "house_loan", "poutcome"}      # cechy modelu docelowego
NAZWY = {"contact": "CONTACT", "age": "AGE", "house_loan": "HOUSING_LOAN",
         "poutcome": "POUTCOME", "education": "EDUCATION", "contact:age": "CONTACT × AGE",
         "contact:house_loan": "CONTACT × HOUSING_LOAN", "contact:education": "CONTACT × EDUCATION"}
p_r = pd.concat([t3_r["p"], t3_r_int["p"]])
p_y = pd.concat([t3_y["p"], t3_y_int["p"]])
rows = []
for el, nazwa in NAZWY.items():
    z_r, z_y = p_r[el] < ALFA, p_y[el] < ALFA
    rl = rola(z_r, z_y)
    if el in W_DOC:
        decyzja = "włączyć"
    elif rl in ("usuwa obciążenie", "poprawia precyzję"):
        decyzja = "włączyć"
    else:
        decyzja = "pominąć"
    rows.append([nazwa, "tak" if el in W_DOC else "nie", p_r[el], p_y[el],
                 "tak" if z_r else "nie", "tak" if z_y else "nie", rl, decyzja])
t_dec = pd.DataFrame(rows, columns=["element", "w modelu docelowym", "p (brak Y)", "p (Y)",
                                    "związek z brakiem", "związek z Y", "rola", "decyzja"])
print(t_dec.round(4).to_string(index=False))
t_dec.round(4).to_csv(OUT / "s2_t11_tabela_decyzyjna.csv", index=False)

# Projekt modelu imputacyjnego wynikający z tabeli decyzyjnej
wlaczone = [NAZWY[e] for e, dec in zip(NAZWY, t_dec["decyzja"]) if dec.startswith("włączyć")]
spec = pd.DataFrame({
    "element": ["zmienna imputowana", "wzorzec braków", "metoda", "zmienne w modelu imputacyjnym",
                "liczba imputacji", "ziarno", "łączenie wyników"],
    "propozycja": [
        "Y_NEWDEPOSIT (jedyna zmienna z brakami: 2 879 obserwacji, 25,47%)",
        "monotoniczny (brakuje tylko Y)",
        "regresja logistyczna dla zmiennej skokowej (wzorzec monotoniczny)",
        ", ".join(wlaczone),
        "m = 500; efektywność (1 + γ/m)^(-1) sprawdzić po imputacji",
        str(SEED),
        "model docelowy w każdym zbiorze, wyniki łączone regułami Rubina"]})
print("\nProjekt modelu imputacyjnego:\n", spec.to_string(index=False))
spec.to_csv(OUT / "s2_t12_projekt_modelu_imputacyjnego.csv", index=False)
print("\nPliki zapisane w:", OUT)


print("\n" + "=" * 100 + "\nSEKCJA 3\n" + "=" * 100)
# =============================================================================
# SEKCJA 3. Punkt 3: model regresji logistycznej przy założeniach MCAR oraz MAR
# Lokalizacja danych i folderu wyników jest zdefiniowana w sekcji 0
# Pliki wynikowe tej części mają przedrostek s3_.
# =============================================================================

pd.set_option("display.width", 160)
print("Wersje: Python", sys.version.split()[0], "| pandas", pd.__version__,
      "| numpy", np.__version__, "| scipy", scipy.__version__,
      "| statsmodels", statsmodels.__version__)

d = load_clean()
obs = d[d["r"] == 0]  # tylko klienci z obserwowanym Y
OUT = WYNIKI

# Model docelowy z zadania. Kategorie referencyjne (z sekcji 0):
# contact = 2 (kanał B), house_loan = yes, poutcome = unknown
F_DOCELOWY = "y ~ contact + age + house_loan + poutcome"

# -----------------------------------------------------------------------------
# 1. MCAR - analiza kompletnych przypadków
# -----------------------------------------------------------------------------
print("\n### 1. Model docelowy przy MCAR (kompletne przypadki)")
m_cc = smf.logit(F_DOCELOWY, data=obs).fit(disp=0)
print(m_cc.summary())

# Przy ref = 2 współczynnik contact[T.1] to od razu beta1 - beta2 (A wobec B)
est_cc = m_cc.params["contact[T.1]"]
se_cc = m_cc.bse["contact[T.1]"]
ci_cc = m_cc.conf_int().loc["contact[T.1]"]
print(f"\nbeta1 - beta2 = {est_cc:.4f}, SE = {se_cc:.4f}, p = {m_cc.pvalues['contact[T.1]']:.4f}")
print(f"OR A wobec B = {np.exp(est_cc):.4f}, 95% PU: {np.exp(ci_cc[0]):.4f} - {np.exp(ci_cc[1]):.4f}")

# Sprawdzenie: model jak we wzorze z zadania (ref = 0, brak komunikacji),
# beta1 - beta2 liczone ręcznie. Powinno wyjść to samo co wyżej.
m_cc0 = smf.logit("y ~ C(contact, Treatment('0')) + age + house_loan + poutcome",
                  data=obs).fit(disp=0)
b1 = "C(contact, Treatment('0'))[T.1]"
b2 = "C(contact, Treatment('0'))[T.2]"
V = m_cc0.cov_params()
est_cc0 = m_cc0.params[b1] - m_cc0.params[b2]
se_cc0 = np.sqrt(V.loc[b1, b1] + V.loc[b2, b2] - 2 * V.loc[b1, b2])
print(f"Sprawdzenie (ref = 0): beta1 - beta2 = {est_cc0:.4f}, SE = {se_cc0:.4f}")
assert abs(est_cc0 - est_cc) < 1e-6 and abs(se_cc0 - se_cc) < 1e-6, "Kontrast niezgodny"

# -----------------------------------------------------------------------------
# 2. MAR - model imputacyjny (specyfikacja z punktu 2)
# -----------------------------------------------------------------------------
print("\n### 2. Model imputacyjny dla Y")
F_IMP = "contact + age + house_loan + poutcome + education + contact:house_loan"
X = dmatrix(F_IMP, d, return_type="dataframe")  # macierz cech dla wszystkich klientów
braki = (d["r"] == 1).to_numpy()

m_imp = sm.Logit(d.loc[~braki, "y"], X[~braki]).fit(disp=0)
print(m_imp.summary())

# Macierz cech modelu docelowego (te same kolumny dla każdego uzupełnionego zbioru)
X_DOC = dmatrix("contact + age + house_loan + poutcome", d, return_type="dataframe")
K_AB = list(X_DOC.columns).index("contact[T.1]")      # beta1 - beta2

M = 500                                               # liczba imputacji
rng = np.random.default_rng(SEED)


def imputuj():
    """Jedna imputacja: losujemy parametry modelu, a potem brakujące Y.
    method='cholesky' daje te same liczby losowe na każdym komputerze."""
    beta = rng.multivariate_normal(m_imp.params, m_imp.cov_params(), method="cholesky")
    p = 1 / (1 + np.exp(-(X.to_numpy()[braki] @ beta)))
    y_imp = d["y"].to_numpy(dtype=float).copy()
    y_imp[braki] = rng.binomial(1, p)
    return beta, y_imp


# -----------------------------------------------------------------------------
# 3. MAR - M imputacji, model docelowy w każdej, łączenie regułami Rubina
# -----------------------------------------------------------------------------
print(f"\n### 3. Imputacja wielokrotna (M = {M})")
oceny, bledy, odsetki, imputacje, parametry = [], [], [], [], []
kanal = d["contact"].astype(str).to_numpy()
for i in range(M):
    beta, y_imp = imputuj()
    parametry.append(beta)
    imputacje.append(y_imp[braki].astype(int))
    m = sm.Logit(y_imp, X_DOC).fit(disp=0)
    oceny.append(m.params.iloc[K_AB])
    bledy.append(m.bse.iloc[K_AB])
    odsetki.append({k: y_imp[braki & (kanal == k)].mean() for k in ["2", "0", "1"]})

# Sprawdzenie: odsetek Y = 1 według kanału, obserwowane vs imputowane (średnio z M)
spr = pd.DataFrame({
    "obserwowane Y": obs.groupby("contact", observed=True)["y"].mean(),
    "imputowane Y": pd.DataFrame(odsetki).mean()})
print("Odsetek Y = 1 według kanału:")
print(spr.round(4))

# Reguły Rubina
oceny = np.array(oceny)
bledy = np.array(bledy)
Q = oceny.mean()                              # ocena punktowa
W = (bledy ** 2).mean()                       # wariancja wewnątrzimputacyjna
B = oceny.var(ddof=1)                         # wariancja międzyimputacyjna
T = W + (1 + 1 / M) * B                       # wariancja całkowita
se_mi = np.sqrt(T)                            # błąd standardowy
gamma = (1 + 1 / M) * B / W                   # względny wzrost wariancji
v = (M - 1) * (1 + 1 / gamma) ** 2            # stopnie swobody
fmi = (gamma + 2 / (v + 3)) / (gamma + 1)     # frakcja brakującej informacji
efekt = 1 / (1 + fmi / M)                     # efektywność

t_kr = stats.t.ppf(0.975, v)
ci_mi = (Q - t_kr * se_mi, Q + t_kr * se_mi)
p_mi = 2 * stats.t.sf(abs(Q / se_mi), v)

print(f"\nbeta1 - beta2 = {Q:.4f}, SE = {se_mi:.4f}, p = {p_mi:.4f}")
print(f"wariancja wewnątrz = {W:.5f}, między = {B:.5f}, całkowita = {T:.5f}")
print(f"wzrost wariancji = {gamma:.4f}, df = {v:.1f}, FMI = {fmi:.4f}, efektywność = {efekt:.4f}")
print(f"OR A wobec B = {np.exp(Q):.4f}, 95% PU: {np.exp(ci_mi[0]):.4f} - {np.exp(ci_mi[1]):.4f}")

# -----------------------------------------------------------------------------
# 4. Porównanie MCAR i MAR
# -----------------------------------------------------------------------------
print("\n### 4. Porównanie wyników")
wyniki = pd.DataFrame([
    {"metoda": "Kompletne przypadki (MCAR)", "beta1 - beta2": est_cc, "błąd": se_cc,
     "OR A wobec B": np.exp(est_cc), "OR dolna 95%": np.exp(ci_cc[0]),
     "OR górna 95%": np.exp(ci_cc[1]), "p": m_cc.pvalues["contact[T.1]"], "FMI": np.nan},
    {"metoda": "Imputacja wielokrotna (MAR)", "beta1 - beta2": Q, "błąd": se_mi,
     "OR A wobec B": np.exp(Q), "OR dolna 95%": np.exp(ci_mi[0]),
     "OR górna 95%": np.exp(ci_mi[1]), "p": p_mi, "FMI": fmi},
])
print(wyniki.round(4).to_string(index=False))

# -----------------------------------------------------------------------------
# 5. Zapis wyników (tabele do raportu oraz imputacje do diagnostyki w punkcie 5)
# -----------------------------------------------------------------------------
wyniki.to_csv(OUT / "s3_tabela_wynikow.csv", index=False)

tab_cc = pd.DataFrame({"beta": m_cc.params, "SE": m_cc.bse, "p": m_cc.pvalues,
                       "OR": np.exp(m_cc.params)})
tab_cc.to_csv(OUT / "s3_model_mcar.csv")
pd.DataFrame({"wartość": [m_cc0.params[b1], m_cc0.params[b2], V.loc[b1, b2], est_cc0, se_cc0]},
             index=["beta1 (A wobec braku komunikacji)", "beta2 (B wobec braku komunikacji)",
                    "Cov(beta1, beta2)", "beta1 - beta2", "SE(beta1 - beta2)"]
             ).to_csv(OUT / "s3_kontrola_kontrastu.csv")

rubin = pd.DataFrame({"miara": ["m (liczba imputacji)", "ocena beta1 - beta2",
                                "wariancja wewnątrz imputacji", "wariancja między imputacjami",
                                "wariancja całkowita", "błąd standardowy",
                                "względny wzrost wariancji", "stopnie swobody", "FMI",
                                "efektywność"],
                      "wartość": [M, Q, W, B, T, se_mi, gamma, v, fmi, efekt]})
rubin.to_csv(OUT / "s3_reguly_rubina.csv", index=False)
spr.to_csv(OUT / "s3_odsetek_y_imputowane.csv")

# wynik z każdej imputacji osobno (ocena i błąd beta1 - beta2)
pd.DataFrame({"imputacja": range(1, M + 1), "beta1 - beta2": oceny,
              "błąd": bledy}).to_csv(OUT / "s3_oceny_imputacji.csv", index=False)
# wylosowane parametry modelu imputacyjnego w każdej imputacji (potrzebne do PPC w punkcie 5)
pd.DataFrame(np.vstack(parametry), columns=X.columns,
             index=pd.Index(range(1, M + 1), name="imputacja")).to_csv(OUT / "s3_parametry_imputacji.csv")
# imputowane wartości Y: wiersz = klient z brakiem Y (indeks z dane_czyste), kolumna = imputacja
pd.DataFrame(np.column_stack(imputacje), index=d.index[braki],
             columns=[f"imp_{i}" for i in range(1, M + 1)]).to_csv(OUT / "s3_imputacje.csv")
print("\nZapisano pliki s3_*.csv w", OUT)


print("\n" + "=" * 100 + "\nSEKCJA 4\n" + "=" * 100)
# =============================================================================
# SEKCJA 4. Punkt 5 i aspekt innowacyjny: diagnostyka imputacji i analiza wrażliwości
# Lokalizacja danych i folderu wyników jest zdefiniowana w sekcji 0.
# Diagnostyka korzysta z tych samych imputacji co sekcja 3
# (pliki s3_imputacje.csv i s3_parametry_imputacji.csv), więc wynik MAR jest jeden.
# Pliki wynikowe tej części mają przedrostek s4_.
#
#   1. Miary efektywności imputacji (względny wzrost wariancji, FMI, efektywność)
#   2. Proponowana miara diagnostyczna: PPC dla beta1 - beta2
#   3. Czy wynik zależy od modelu imputacyjnego? (dwa warianty modelu)
#   4. Druga metoda korekty przy MAR: ważenie odwrotnością prawdopodobieństwa obserwacji (IPW)
#   5. Analiza wrażliwości na braki nielosowe (MNAR): przesunięcie delta, punkt przełamania
#   6. Wspólna tabela wyników
# =============================================================================

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
OUT = WYNIKI

print("Wersje: Python", sys.version.split()[0], "| pandas", pd.__version__,
      "| numpy", np.__version__, "| scipy", scipy.__version__,
      "| statsmodels", statsmodels.__version__, "| matplotlib", matplotlib.__version__)

# Wygląd wykresów taki sam jak w sekcjach 1 i 2
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})
NIEBIESKI, POMARANCZ, SZARY = "#1f77b4", "#ff7f0e", "#8c8c8c"


def pl(x, n=2):
    """Liczba z przecinkiem dziesiętnym (zapis polski)."""
    return f"{x:.{n}f}".replace(".", ",")


def os_pl(ax, os="y"):
    """Przecinek dziesiętny na podziałce osi."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.xaxis if os == "x" else ax.yaxis).set_major_formatter(fmt)


def save(name, rect=(0, 0, 1, 1)):
    """Zapisuje bieżący wykres do folderu wyniki/ i zamyka go."""
    plt.tight_layout(rect=rect)
    plt.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close()


# =============================================================================
# 0. DANE, MODELE I NARZĘDZIA
# =============================================================================
d = load_clean()
y = d["y"].to_numpy(dtype=float)
braki = (d["r"] == 1).to_numpy()
kanal = d["contact"].astype(str).to_numpy()            # "0", "1" (A), "2" (B)
isA, isB = kanal == "1", kanal == "2"

# Model docelowy (te same kolumny co w sekcji 3); contact[T.1] = beta1 - beta2
X_DOC = dmatrix("contact + age + house_loan + poutcome", d, return_type="dataframe")
K_AB = list(X_DOC.columns).index("contact[T.1]")
# Model imputacyjny z punktu 2 (ten sam co w sekcji 3) i jego warianty
MODELE_IMP = {
    "główny": "contact + age + house_loan + poutcome + education + contact:house_loan",
    "bez interakcji CONTACT × HOUSING_LOAN": "contact + age + house_loan + poutcome + education",
    "z dodaną interakcją CONTACT × AGE": "contact + age + house_loan + poutcome + education"
                                         " + contact:house_loan + contact:age",
    "celowo błędny (bez CONTACT)": "age + house_loan + poutcome + education"}


def kontrast(yv):
    """Model docelowy na danych yv -> (beta1 - beta2, jego wariancja)."""
    m = sm.Logit(yv, X_DOC).fit(disp=0)
    return m.params.iloc[K_AB], m.cov_params().iloc[K_AB, K_AB]


def rubin(oceny, wariancje):
    """Reguły Rubina (jak w sekcji 3) -> słownik z oceną, błędem, przedziałem, p, FMI."""
    oceny, wariancje = np.asarray(oceny), np.asarray(wariancje)
    m = len(oceny)
    Q, W, B = oceny.mean(), wariancje.mean(), oceny.var(ddof=1)
    T = W + (1 + 1 / m) * B
    riv = (1 + 1 / m) * B / W                             # względny wzrost wariancji
    df = (m - 1) * (1 + 1 / riv) ** 2
    fmi = (riv + 2 / (df + 3)) / (riv + 1)
    se, tkr = np.sqrt(T), stats.t.ppf(.975, (m - 1) * (1 + 1 / riv) ** 2)
    return {"ocena": Q, "błąd": se, "dolna": Q - tkr * se, "górna": Q + tkr * se,
            "p": 2 * stats.t.sf(abs(Q / se), df), "FMI": fmi, "m": m}


def wiersz(metoda, r):
    """Wiersz wspólnej tabeli wyników (format jak s3_tabela_wynikow.csv)."""
    return {"metoda": metoda, "beta1 - beta2": r["ocena"], "błąd": r["błąd"],
            "OR A wobec B": np.exp(r["ocena"]), "OR dolna 95%": np.exp(r["dolna"]),
            "OR górna 95%": np.exp(r["górna"]), "p": r["p"], "FMI": r.get("FMI", np.nan)}


def imputacja_wielokrotna(model, M, seed, deltaA=0.0, deltaB=0.0, replikuj=False):
    """M imputacji brakujących Y regresją logistyczną z losowaniem parametrów (jak w
    sekcji 3) i model docelowy w każdej. deltaA, deltaB: przesunięcie logitu szansy
    zakupu klientów bez Y w kanale A i B (analiza MNAR; 0 = MAR). replikuj: losuje Y
    także dla obserwowanych Y (dane replikowane do PPC)."""
    X = dmatrix(MODELE_IMP[model], d, return_type="dataframe").to_numpy()
    fit = sm.Logit(y[~braki], X[~braki]).fit(disp=0)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(M):
        beta = rng.multivariate_normal(fit.params, fit.cov_params(), method="cholesky")
        eta = X @ beta + deltaA * isA + deltaB * isB
        y_uzup = y.copy()
        y_uzup[braki] = rng.binomial(1, expit(eta[braki]))
        est, var = kontrast(y_uzup)
        w = {"ocena": est, "wariancja": var}
        if replikuj:
            y_rep = y_uzup.copy()
            y_rep[~braki] = rng.binomial(1, expit(X[~braki] @ beta))
            w["ocena replikowana"] = kontrast(y_rep)[0]
        rows.append(w)
    return pd.DataFrame(rows)


wyniki = []                                    # wiersze wspólnej tabeli wyników

# =============================================================================
# 1. MIARY EFEKTYWNOŚCI IMPUTACJI (z reguł Rubina w sekcji 3)
# =============================================================================
print("\n### 1. Miary efektywności imputacji wielokrotnej (M z sekcji 3)")
rub = pd.read_csv(OUT / "s3_reguly_rubina.csv", index_col=0)["wartość"]
M = int(rub["m (liczba imputacji)"])
t_efekt = pd.DataFrame({"miara": ["liczba imputacji m", "względny wzrost wariancji",
                                  "frakcja brakującej informacji (FMI)", "efektywność (1 + FMI/m)^(-1)"],
                        "wartość": [M, rub["względny wzrost wariancji"], rub["FMI"], rub["efektywność"]]})
print(t_efekt.round(4).to_string(index=False))
t_efekt.round(4).to_csv(OUT / "s4_t1_miary_efektywnosci.csv", index=False)

# =============================================================================
# 2. PROPONOWANA MIARA: PPC DLA beta1 - beta2
# =============================================================================
# Dane replikowane: w każdej imputacji z sekcji 3, z tymi samymi wylosowanymi
# parametrami modelu imputacyjnego, losujemy Y także dla klientów z obserwowanym Y.
# PPC p = odsetek imputacji, w których beta1 - beta2 z danych replikowanych jest
# większe niż z danych uzupełnionych. Wartości bliskie 0,5: model odtwarza kontrast;
# bliskie 0 lub 1: model imputacyjny systematycznie zaniża lub zawyża kontrast.
print("\n### 2. Predykcyjna wartość p a posteriori (PPC) dla beta1 - beta2")
param = pd.read_csv(OUT / "s3_parametry_imputacji.csv", index_col=0)
imp = pd.read_csv(OUT / "s3_imputacje.csv", index_col=0)
oceny_s3 = pd.read_csv(OUT / "s3_oceny_imputacji.csv")
X_GL = dmatrix(MODELE_IMP["główny"], d, return_type="dataframe")
assert list(X_GL.columns) == list(param.columns), "Kolumny modelu imputacyjnego niezgodne z sekcją 3"
X_GL = X_GL.to_numpy()
rng = np.random.default_rng(SEED + 1)
rows = []
for i in range(M):
    y_uzup = y.copy()
    y_uzup[braki] = imp[f"imp_{i + 1}"].to_numpy()
    y_rep = y_uzup.copy()
    y_rep[~braki] = rng.binomial(1, expit(X_GL[~braki] @ param.iloc[i].to_numpy()))
    rows.append({"ocena": oceny_s3["beta1 - beta2"].iloc[i], "ocena replikowana": kontrast(y_rep)[0],
                 **{f"Y=1 uzupełnione {k}": y_uzup[kanal == c].mean() for k, c in [("A", "1"), ("B", "2"), ("0", "0")]},
                 **{f"Y=1 replikowane {k}": y_rep[kanal == c].mean() for k, c in [("A", "1"), ("B", "2"), ("0", "0")]}})
ppc_gl = pd.DataFrame(rows)
# kontrola: ocena z danych uzupełnionych jest taka sama jak w sekcji 3
y0 = y.copy()
y0[braki] = imp["imp_1"].to_numpy()
assert abs(kontrast(y0)[0] - ppc_gl["ocena"].iloc[0]) < 1e-8, "Imputacje niezgodne z sekcją 3"


def ppc_tabela(df, z_odsetkami=True):
    stat = [("beta1 - beta2 (A wobec B)", "ocena", "ocena replikowana")]
    if z_odsetkami:
        stat += [(f"odsetek Y=1, kanał {k}", f"Y=1 uzupełnione {k}", f"Y=1 replikowane {k}") for k in "AB0"]
    return pd.DataFrame([{"statystyka": s, "średnia uzupełnione": df[a].mean(),
                          "średnia replikowane": df[b].mean(), "PPC p": (df[b] > df[a]).mean()}
                         for s, a, b in stat])


# Kontrola czułości miary: celowo błędny model imputacyjny bez CONTACT
M_ZLY = 100
ppc_zly = imputacja_wielokrotna("celowo błędny (bez CONTACT)", M_ZLY, SEED + 2, replikuj=True)
t_ppc = pd.concat({f"model z punktu 2 (M = {M})": ppc_tabela(ppc_gl).set_index("statystyka"),
                   f"model celowo błędny, bez CONTACT (M = {M_ZLY})":
                       ppc_tabela(ppc_zly, False).set_index("statystyka")})
print(t_ppc.round(4))
t_ppc.round(4).to_csv(OUT / "s4_t2_ppc.csv")

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
for ax, (tyt, df, kol) in zip(axs, [("Model imputacyjny z punktu 2", ppc_gl, NIEBIESKI),
                                     ("Model celowo błędny (bez CONTACT)", ppc_zly, POMARANCZ)]):
    a, b = df["ocena"], df["ocena replikowana"]
    lo, hi = min(a.min(), b.min()) - .05, max(a.max(), b.max()) + .05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.scatter(a, b, s=14, alpha=.6, color=kol)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="β1 − β2 na danych uzupełnionych",
           ylabel="β1 − β2 na danych replikowanych", title=f"{tyt}\nPPC p = {pl((b > a).mean())}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=.3)
    os_pl(ax, "x")
    os_pl(ax)
fig.text(.01, .01, "Każdy punkt to jedna imputacja. Chmura po obu stronach przekątnej oznacza, "
         "że model imputacyjny odtwarza kontrast A wobec B.", fontsize=8, color="0.3")
save("s4_w1_ppc.png", rect=(0, .05, 1, 1))

# =============================================================================
# 3. CZY WYNIK ZALEŻY OD MODELU IMPUTACYJNEGO?
# =============================================================================
print("\n### 3. Warianty modelu imputacyjnego")
s3 = pd.read_csv(OUT / "s3_tabela_wynikow.csv")
rows = [{"wariant": "model z punktu 2 (główny)", "beta1 - beta2": s3.iloc[1]["beta1 - beta2"],
         "błąd": s3.iloc[1]["błąd"], "OR A wobec B": s3.iloc[1]["OR A wobec B"], "p": s3.iloc[1]["p"],
         "FMI": s3.iloc[1]["FMI"], "PPC p": t_ppc.iloc[0]["PPC p"]}]
for k, war in enumerate(["bez interakcji CONTACT × HOUSING_LOAN", "z dodaną interakcją CONTACT × AGE"]):
    df_w = imputacja_wielokrotna(war, M, SEED + 10 + k, replikuj=True)
    r = rubin(df_w["ocena"], df_w["wariancja"])
    rows.append({"wariant": war, "beta1 - beta2": r["ocena"], "błąd": r["błąd"],
                 "OR A wobec B": np.exp(r["ocena"]), "p": r["p"], "FMI": r["FMI"],
                 "PPC p": (df_w["ocena replikowana"] > df_w["ocena"]).mean()})
    wyniki.append(wiersz(f"Imputacja wielokrotna (MAR), {war}", r))
t_war = pd.DataFrame(rows)
print(t_war.round(4).to_string(index=False))
t_war.round(4).to_csv(OUT / "s4_t3_warianty_imputacji.csv", index=False)

# =============================================================================
# 4. IPW: WAŻENIE ODWROTNOŚCIĄ PRAWDOPODOBIEŃSTWA OBSERWACJI Y
# =============================================================================
print("\n### 4. IPW")
# Wagi 1 / P(Y obserwowane | X) z modelu braku z punktu 2 (s2_p_obserwacji.csv),
# stabilizowane odsetkiem obserwowanych Y. Model docelowy na obserwowanych Y z wagami,
# błędy standardowe odporne (kanapkowe), bo wagi nie są liczebnościami.
p2 = pd.read_csv(OUT / "s2_p_obserwacji.csv")
assert (p2["r"].to_numpy() == d["r"].to_numpy()).all(), "Niezgodny plik s2_p_obserwacji.csv"
wagi = (1 - braki.mean()) / p2["p_obs"].to_numpy()[~braki]
m_ipw = sm.GLM(y[~braki], X_DOC[~braki], family=sm.families.Binomial(),
               var_weights=wagi).fit(cov_type="HC0")
est, se = m_ipw.params.iloc[K_AB], m_ipw.bse.iloc[K_AB]
r_ipw = {"ocena": est, "błąd": se, "dolna": est - 1.959964 * se, "górna": est + 1.959964 * se,
         "p": 2 * stats.norm.sf(abs(est) / se)}
wyniki.append(wiersz("IPW (MAR), wagi z modelu braku z punktu 2", r_ipw))
t_wagi = pd.DataFrame([{"min": wagi.min(), "Q1": np.percentile(wagi, 25), "mediana": np.median(wagi),
                        "Q3": np.percentile(wagi, 75), "max": wagi.max(), "max/min": wagi.max() / wagi.min()}])
print("Rozkład wag:\n", t_wagi.round(3).to_string(index=False))
print(f"IPW: beta1 - beta2 = {est:.4f}, SE = {se:.4f}, OR = {np.exp(est):.4f}")
t_wagi.round(4).to_csv(OUT / "s4_t4_ipw_wagi.csv", index=False)

# =============================================================================
# 5. ANALIZA WRAŻLIWOŚCI MNAR (przesunięcie delta)
# =============================================================================
print("\n### 5. Analiza wrażliwości MNAR")
# Klienci bez Y mają logit szansy zakupu przesunięty o delta względem podobnych klientów
# z Y: deltaA w kanale A, deltaB w kanale B (w kanale 0 delta = 0). exp(delta) mówi,
# ile razy szansa zakupu klienta bez Y różni się od szansy podobnego klienta z Y.
M_MNAR = 50

# 5a. Linie: przesunięcie tylko w kanale A albo tylko w kanale B, krok 0,25
KROK = np.round(np.arange(-2, 2.01, .25), 2)
rows = []
for j, (nazwa, ka, kb) in enumerate([("tylko kanał A (δB = 0)", 1, 0), ("tylko kanał B (δA = 0)", 0, 1)]):
    for k, t in enumerate(KROK):
        df_m = imputacja_wielokrotna("główny", M_MNAR, SEED + 100 + 50 * j + k, ka * t, kb * t)
        rows.append({"kierunek": nazwa, "delta": t, **wiersz("", rubin(df_m["ocena"], df_m["wariancja"]))})
t_linie = pd.DataFrame(rows).drop(columns="metoda")
t_linie["istotne"] = (t_linie["OR dolna 95%"] > 1) | (t_linie["OR górna 95%"] < 1)
t_linie.round(4).to_csv(OUT / "s4_t5_mnar_linie.csv", index=False)

# Punkt przełamania: najmniejsze |delta| w każdą stronę, przy którym 95% PU OR przestaje zawierać 1
rows = []
for nazwa, g in t_linie.groupby("kierunek", sort=False):
    for znak, strona in [(1, "delta > 0"), (-1, "delta < 0")]:
        gg = g[np.sign(g["delta"]) == znak].sort_values("delta", key=np.abs)
        ist = gg[gg["istotne"]]
        if len(ist):
            p0 = ist.iloc[0]
            rows.append({"kierunek": nazwa, "strona": strona, "delta": p0["delta"],
                         "exp(delta)": np.exp(p0["delta"]), "OR A wobec B": p0["OR A wobec B"],
                         "wniosek": "A istotnie lepszy" if p0["OR dolna 95%"] > 1 else "B istotnie lepszy"})
        else:
            rows.append({"kierunek": nazwa, "strona": strona, "delta": np.nan, "exp(delta)": np.nan,
                         "OR A wobec B": np.nan, "wniosek": "brak przełamania dla |delta| ≤ 2"})
t_przel = pd.DataFrame(rows)
print(f"Punkty przełamania (M = {M_MNAR} na punkt, krok 0,25):\n", t_przel.round(3).to_string(index=False))
t_przel.round(4).to_csv(OUT / "s4_t6_punkt_przelamania.csv", index=False)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (nazwa, g) in zip(axs, t_linie.groupby("kierunek", sort=False)):
    ax.fill_between(g["delta"], g["OR dolna 95%"], g["OR górna 95%"], color=NIEBIESKI, alpha=.2,
                    label="95% przedział ufności")
    ax.plot(g["delta"], g["OR A wobec B"], color=NIEBIESKI, lw=1.8, label="OR A wobec B")
    ax.axhline(1, color="k", lw=1)
    ax.axvline(0, color=SZARY, ls=":", lw=1.5, label="MAR (δ = 0)")
    for _, p0 in t_przel[(t_przel["kierunek"] == nazwa) & t_przel["delta"].notna()].iterrows():
        ax.axvline(p0["delta"], color=POMARANCZ, ls="--", lw=1.2)
    ax.set_yscale("log")
    ax.set_yticks([.5, .75, 1, 1.5, 2, 3])
    ax.set_yticklabels(["0,5", "0,75", "1", "1,5", "2", "3"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set(title=nazwa, xlabel="δ (przesunięcie logitu szansy zakupu klientów bez Y)")
    ax.grid(True, alpha=.3)
    os_pl(ax, "x")
axs[0].set_ylabel("OR zakupu A wobec B")
axs[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.text(.01, .01, "Pomarańczowa linia przerywana: punkt przełamania (95% PU przestaje zawierać 1). "
         f"Każdy punkt: {M_MNAR} imputacji, reguły Rubina.", fontsize=8, color="0.3")
save("s4_w2_mnar_linie.png", rect=(0, .05, 1, 1))

# 5b. Mapa: siatka deltaA x deltaB od -2 do 2 co 0,5
M_SIATKA = 20
SIATKA = np.round(np.arange(-2, 2.01, .5), 2)
rows = []
for k, (dA, dB) in enumerate([(a, b) for a in SIATKA for b in SIATKA]):
    df_m = imputacja_wielokrotna("główny", M_SIATKA, SEED + 1000 + k, dA, dB)
    rows.append({"deltaA": dA, "deltaB": dB, **wiersz("", rubin(df_m["ocena"], df_m["wariancja"]))})
t_siatka = pd.DataFrame(rows).drop(columns="metoda")
t_siatka["istotne"] = (t_siatka["OR dolna 95%"] > 1) | (t_siatka["OR górna 95%"] < 1)
t_siatka.round(4).to_csv(OUT / "s4_t7_mnar_siatka.csv", index=False)
print(f"Siatka {len(SIATKA)} x {len(SIATKA)}, M = {M_SIATKA}: istotnych punktów "
      f"{int(t_siatka['istotne'].sum())} z {len(t_siatka)}")

piv = t_siatka.pivot(index="deltaB", columns="deltaA", values="OR A wobec B").sort_index(ascending=False)
piv_ist = t_siatka.pivot(index="deltaB", columns="deltaA", values="istotne").sort_index(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6.4))
lg = np.log(piv.values)
im = ax.imshow(lg, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=lg.min(), vmax=lg.max()))
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v, ist = piv.iat[i, j], piv_ist.iat[i, j]
        ax.text(j, i, pl(v) + ("*" if ist else ""), ha="center", va="center", fontsize=8,
                fontweight="bold" if ist else "normal", color="white" if abs(np.log(v)) > .45 else "black")
ax.add_patch(plt.Rectangle((list(piv.columns).index(0) - .5, list(piv.index).index(0) - .5),
                           1, 1, fill=False, ec="k", lw=2))
ax.set_xticks(range(len(piv.columns)))
ax.set_xticklabels([pl(v, 1) for v in piv.columns])
ax.set_yticks(range(len(piv.index)))
ax.set_yticklabels([pl(v, 1) for v in piv.index])
ax.set(xlabel="δA: przesunięcie w kanale A", ylabel="δB: przesunięcie w kanale B",
       title="Analiza wrażliwości MNAR: OR zakupu A wobec B")
cb = fig.colorbar(im, ax=ax, shrink=.8)
cb.set_label("OR A wobec B (skala logarytmiczna)")
cb.set_ticks(np.log([.5, .75, 1, 1.5, 2]))
cb.set_ticklabels(["0,5", "0,75", "1", "1,5", "2"])
fig.text(.01, .01, "Czarna ramka: MAR (δA = δB = 0). Pogrubienie i *: 95% przedział ufności OR nie "
         f"zawiera 1. Każde pole: {M_SIATKA} imputacji.", fontsize=8, color="0.3")
save("s4_w3_mnar_mapa.png", rect=(0, .04, 1, 1))

# 5c. Dwa scenariusze umiarkowanego MNAR do wspólnej tabeli (exp(0,5) = 1,65 raza)
M_SCEN = 100
for k, (dA, dB, nazwa) in enumerate([(.5, -.5, "MNAR: δA = 0,5, δB = −0,5"),
                                      (-.5, .5, "MNAR: δA = −0,5, δB = 0,5")]):
    df_m = imputacja_wielokrotna("główny", M_SCEN, SEED + 9000 + k, dA, dB)
    wyniki.append(wiersz(nazwa, rubin(df_m["ocena"], df_m["wariancja"])))

# =============================================================================
# 6. WSPÓLNA TABELA WYNIKÓW (wiersze dopisywane do tabeli z sekcji 3 w sekcji 6)
# =============================================================================
print("\n### 6. Tabela wyników tej części")
t_wyn = pd.DataFrame(wyniki)
print(t_wyn.round(4).to_string(index=False))
t_wyn.to_csv(OUT / "s4_tabela_wynikow.csv", index=False)
print("\nPliki s4_* zapisane w:", OUT)


print("\n" + "=" * 100 + "\nSEKCJA 5\n" + "=" * 100)
# =============================================================================
# SEKCJA 5. Kontrola krzyżowa: tabele z sekcji 1 (wyniki/t*.csv) liczone od nowa z danych
# Każda kontrola porównuje liczbę z tabeli z liczbą policzoną od nowa z danych.
# Wynik: wyniki/s4_recenzja_osoby1.csv (kolumna "zgodne" = tak/nie).
# =============================================================================

W = WYNIKI
d = load_clean()
N = len(d)
kontrole = []


def sprawdz(opis, z_tabeli, z_danych, tol=0.006):
    """Zapisuje jedną kontrolę; tol pozwala na zaokrąglenia w tabelach (2 miejsca)."""
    ok = bool(np.all(np.abs(np.asarray(z_tabeli, float) - np.asarray(z_danych, float)) <= tol))
    kontrole.append({"kontrola": opis, "z tabeli": z_tabeli, "z danych": z_danych,
                     "zgodne": "tak" if ok else "NIE"})


# t1: statystyki wieku
t1 = pd.read_csv(W / "t1_age_opis.csv", index_col=0)
sprawdz("t1: n ogółem = liczba klientów", t1.loc["Ogółem", "n"], N)
sprawdz("t1: suma n w kanałach = n ogółem", t1.filter(like="CONTACT", axis=0)["n"].sum(), N)
sprawdz("t1: suma n wg r = n ogółem", t1.filter(like="r=", axis=0)["n"].sum(), N)
sprawdz("t1: średni wiek ogółem", t1.loc["Ogółem", "średnia"], round(d.age.mean(), 2))
sprawdz("t1: mediana wieku w kanale B", t1.loc["CONTACT B: kanał B", "mediana"],
        d.loc[d.contact == "2", "age"].median())

# t2: cechy skokowe
t2 = pd.read_csv(W / "t2_cechy_skokowe.csv")
for zm, g in t2.groupby("zmienna", sort=False):
    sprawdz(f"t2: {zm}, suma liczności = {N}", g["n"].sum(), N)
    sprawdz(f"t2: {zm}, suma % = 100", g["%"].sum(), 100, tol=0.03)
sprawdz("t2: Y=1", t2.query("zmienna == 'y' and kategoria == '1'")["n"].iloc[0], int((d.y == 1).sum()))
sprawdz("t2: braki Y", t2.query("zmienna == 'y' and kategoria == 'brak (.)'")["n"].iloc[0], int(d.r.sum()))

# t3: tabele krzyżowe z CONTACT
for c in ["house_loan", "education", "poutcome"]:
    t3 = pd.read_csv(W / f"t3_krzyzowa_{c}.csv", header=[0, 1], index_col=0)
    n_kan = t3["n"].sum().to_numpy()
    sprawdz(f"t3 {c}: liczności kanałów (0, A, B)", list(n_kan),
            [int((d.contact == k).sum()) for k in ["0", "1", "2"]])
    sprawdz(f"t3 {c}: suma = {N}", n_kan.sum(), N)
    sprawdz(f"t3 {c}: % w kanale sumują się do 100", list(t3["% w kanale"].sum().round(1)),
            [100, 100, 100], tol=0.15)

# t4: wynik badania na obserwowanych Y
t4 = pd.read_csv(W / "t4_y_wg_contact.csv", index_col=0)
obs = d[d.r == 0]
sprawdz("t4: suma n_obs = liczba obserwowanych Y", t4["n_obs"].sum(), len(obs))
sprawdz("t4: suma Y=1 = liczba zakupów", t4["Y=1"].sum(), int(obs.y.sum()))
sprawdz("t4: odsetek Y=1 w kanale A", round(t4.iloc[1]["odsetek"], 4),
        round(obs.loc[obs.contact == "1", "y"].mean(), 4), tol=1e-4)

# t5: podgrupy (każda cecha osobno sumuje się do obserwowanych Y)
t5 = pd.read_csv(W / "t5_y_podgrupy.csv")
for cecha in ["house_loan", "poutcome"]:
    g = t5[t5.podgrupa.str.startswith(cecha)]
    sprawdz(f"t5 {cecha}: suma n_obs = obserwowane Y", g["n_obs"].sum(), len(obs))
    sprawdz(f"t5 {cecha}: suma Y=1 = zakupy", g["Y=1"].sum(), int(obs.y.sum()))

# t6-t8: braki
t6 = pd.read_csv(W / "t6_braki_zmienne.csv", index_col=0)
sprawdz("t6: braki tylko w Y", t6["braki"].drop("y").sum(), 0)
t7 = pd.read_csv(W / "t7_wzorce_brakow.csv")
sprawdz("t7: wzorce sumują się do N", t7["n"].sum(), N)
t8 = pd.read_csv(W / "t8_braki_wg_contact.csv", index_col=0)
sprawdz("t8: suma braków Y w kanałach", t8["braki Y"].sum(), int(d.r.sum()))
sprawdz("t8: suma n w kanałach", t8["n"].sum(), N)

# t9: SMD wieku A wobec B, przeliczone niezależnie
a, b = d.loc[d.contact == "1", "age"], d.loc[d.contact == "2", "age"]
t9 = pd.read_csv(W / "t9_smd.csv", header=[0, 1], index_col=0)
sprawdz("t9: SMD wieku A wobec B (pełna próba)", t9.loc["age", ("pełna próba", "A vs B")],
        round(abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2), 4), tol=1e-4)

wynik = pd.DataFrame(kontrole)
print(wynik.to_string(index=False))
print(f"\nKontroli: {len(wynik)}, niezgodnych: {(wynik['zgodne'] == 'NIE').sum()}")
wynik.to_csv(W / "s4_recenzja_osoby1.csv", index=False)


print("\n" + "=" * 100 + "\nSEKCJA 6\n" + "=" * 100)
# =============================================================================
# SEKCJA 6. Punkt 4: podsumowanie wyników estymacji modelu docelowego w kontekście braków danych
# Łączy tabele wyników z sekcji 3 i 4 w jedną
# tabelę zbiorczą i rysuje wykres leśny OR A wobec B dla wszystkich metod.
# Pliki wynikowe tej części mają przedrostek s5_.
# =============================================================================

OUT = WYNIKI
pd.set_option("display.width", 160)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})

# 1. TABELA ZBIORCZA: wszystkie metody w jednym formacie
s3 = pd.read_csv(OUT / "s3_tabela_wynikow.csv")
s4 = pd.read_csv(OUT / "s4_tabela_wynikow.csv")
t = pd.concat([s3, s4], ignore_index=True)
# Założenie, przy którym wynik jest poprawny
t.insert(1, "założenie", ["MCAR", "MAR"] + ["MNAR" if m.startswith("MNAR") else "MAR" for m in s4["metoda"]])
t["istotne (5%)"] = np.where((t["OR dolna 95%"] > 1) | (t["OR górna 95%"] < 1), "tak", "nie")
print("### Tabela zbiorcza: OR zakupu A wobec B")
print(t.round(4).to_string(index=False))
t.to_csv(OUT / "s5_tabela_zbiorcza.csv", index=False)

# 2. WYKRES LEŚNY OR A wobec B
KOL = {"MCAR": "#8c8c8c", "MAR": "#1f77b4", "MNAR": "#ff7f0e"}
etykiety = (t["metoda"].str.replace("Imputacja wielokrotna (MAR), ", "MI, ", regex=False)
            .str.replace(" (MAR)", "", regex=False)
            .str.replace("IPW, wagi z modelu braku z punktu 2", "IPW (wagi z modelu braku)", regex=False))
etykiety.iloc[1] = "Imputacja wielokrotna (model główny)"
poz = np.arange(len(t))[::-1]
fig, ax = plt.subplots(figsize=(9, 4.6))
for i, (_, w) in zip(poz, t.iterrows()):
    ax.errorbar(w["OR A wobec B"], i, xerr=[[w["OR A wobec B"] - w["OR dolna 95%"]],
                                             [w["OR górna 95%"] - w["OR A wobec B"]]],
                fmt="o", color=KOL[w["założenie"]], capsize=4, ms=7)
    ax.text(2.35, i, f"{w['OR A wobec B']:.2f} ({w['OR dolna 95%']:.2f}–{w['OR górna 95%']:.2f})"
            .replace(".", ","), va="center", fontsize=8)
ax.axvline(1, color="k", lw=1)
ax.set_xscale("log")
ax.set_xlim(.6, 2.3)
ax.set_xticks([.6, .8, 1, 1.25, 1.5, 2])
ax.set_xticklabels(["0,6", "0,8", "1", "1,25", "1,5", "2"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_yticks(poz)
ax.set_yticklabels(etykiety)
ax.set(xlabel="OR zakupu A wobec B (skala logarytmiczna), 95% przedział ufności",
       title="Iloraz szans zakupu A wobec B według metody")
ax.xaxis.grid(True, alpha=.3)
for zal, kol in KOL.items():
    ax.plot([], [], "o", color=kol, label=f"założenie {zal}")
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(.5, -.16))
fig.text(.01, .01, "OR > 1: szansa zakupu w kanale A większa niż w kanale B. Przedział przecinający "
         "linię OR = 1 oznacza brak istotnej różnicy.", fontsize=8, color="0.3")
plt.tight_layout(rect=(0, .03, 1, 1))
plt.savefig(OUT / "s5_w1_wykres_lesny.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nZapisano s5_tabela_zbiorcza.csv i s5_w1_wykres_lesny.png w", OUT)
