# =============================================================================
# Punkt 1: analiza struktury i braków
# Lokalizacja danych i folderu wyników jest zdefiniowana w 00_setup.py
# =============================================================================
import sys
import importlib
import numpy as np
import pandas as pd
import scipy
import statsmodels
import matplotlib
matplotlib.use("Agg")                    # zapis wykresów do plików, bez okna
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

S = importlib.import_module("00_setup")  # ścieżki, ziarno, load_clean()
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

print("Wersje: Python", sys.version.split()[0], "| pandas", pd.__version__,
      "| numpy", np.__version__, "| scipy", scipy.__version__,
      "| statsmodels", statsmodels.__version__, "| matplotlib", matplotlib.__version__)

# ---- Wygląd wykresów (wspólny dla wszystkich) -------------------------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})

# ---- 0. PRZYGOTOWANIE -------------------------------------------------------
d = S.load_clean()
OUT = S.WYNIKI
LAB = S.CONTACT_LAB
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


# ---- 1. KONTROLA DANYCH -----------------------------------------------------
print("\n### 1. Kontrola danych")
raw = S.load_raw()
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

# ---- 2. CECHY CIĄGŁE (AGE): statystyki opisowe ------------------------------
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

# ---- 3. CECHY SKOKOWE -------------------------------------------------------
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

# ---- 4. WYNIK BADANIA: % Y=1 według CONTACT (obserwowane Y) -----------------
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

# ---- 5. BRAKI DANYCH --------------------------------------------------------
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

# ---- 6. RÓWNOWAGA LOSOWANIA: SMD między grupami CONTACT ---------------------
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
