# =============================================================================
# Punkt 2: mechanizm braków danych w Y
# Czy prawdopodobieństwo braku Y zależy od pozostałych cech i czy są cechy,
# za pomocą których imputacja może skorygować oszacowania?
# Metody jak na wykładzie (slajdy 94 i 103):
#   Metoda I  - test t różnicy średnich wieku między grupami braku (PROC TTEST)
#   Metoda II - test chi-kwadrat i V Cramera dla cech skokowych (PROC FREQ / CHISQ)
#   Model logistyczny prawdopodobieństwa obserwacji Y, P(M = 0) (PROC LOGISTIC)
# Lokalizacja danych i folderu wyników jest zdefiniowana w 00_setup.py.
# Pliki wynikowe tej części mają przedrostek s2_.
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
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import roc_auc_score

S = importlib.import_module("00_setup")  # ścieżki, ziarno, load_clean()
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

print("Wersje: Python", sys.version.split()[0], "| pandas", pd.__version__,
      "| numpy", np.__version__, "| scipy", scipy.__version__,
      "| statsmodels", statsmodels.__version__, "| matplotlib", matplotlib.__version__)

# Wygląd wykresów taki sam jak w 01_struktura.py
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})

# 0. PRZYGOTOWANIE
d = S.load_clean()
OUT = S.WYNIKI
LAB = S.CONTACT_LAB
d["kanal"] = d["contact"].astype(int).map(LAB)
ORDER = [LAB[0], LAB[1], LAB[2]]
COL = {LAB[0]: "#8c8c8c", LAB[1]: "#1f77b4", LAB[2]: "#ff7f0e"}   # kolory kanałów jak u Osoby 1
NIEBIESKI = "#4c72b0"                                              # kolor słupków jak u Osoby 1
# Wskaźnik braku: na wykładzie M (1 = brak). W kodzie grupy nazywa się r (00_setup.py).
# Model logistyczny opisuje, jak na slajdzie 103, prawdopodobieństwo obserwacji Y: P(M = 0).
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
# 1. METODA I: TEST t RÓŻNICY ŚREDNICH WIEKU (slajd 94, PROC TTEST)
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
# 2. METODA II: TEST CHI-KWADRAT DLA CECH SKOKOWYCH (slajd 94, PROC FREQ / CHISQ)
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
# 3. MODEL LOGISTYCZNY PRAWDOPODOBIEŃSTWA OBSERWACJI Y, P(M = 0) (slajd 103)
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

# 3a. Czy cechy działają na brak Y tak samo w każdym kanale? Interakcje z CONTACT
# (jak interakcja j3*lnCost w przykładzie BEEPS2009, slajd 157). Interakcji
# CONTACT x POUTCOME nie badamy: w kanale 0 aż 99,3% klientów ma poutcome = unknown.
INTER = "contact:age + contact:house_loan + contact:education"
m_r_int = logit(f"obs_y ~ {GLOWNE} + {INTER}", d)
t3_r_int = type3(m_r_int).loc[["contact:age", "contact:house_loan", "contact:education"]]
print("\nInterakcje w modelu braku (Wald, Type 3):\n", t3_r_int.round(4))

# Odsetek braków według wieku i kanału (opis interakcji CONTACT x AGE)
t_wk = odsetki(d, "r", ["kanal", "wiek_p"]).reset_index()
print("\nOdsetek braków Y według kanału i wieku:\n", t_wk.round(3).to_string(index=False))
t_wk.round(4).to_csv(OUT / "s2_t6_braki_wiek_kanal.csv", index=False)

# 3b. P(Y zaobserwowane | X) dla każdej obserwacji -> wagi IPW (Osoba 4, slajdy 102-103)
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
efekty_sas = [e.replace(":", "*") for e, dec in zip(NAZWY, t_dec["decyzja"]) if dec.startswith("włączyć")]
spec = pd.DataFrame({
    "element": ["zmienna imputowana", "wzorzec braków", "metoda", "zmienne w modelu imputacyjnym",
                "liczba imputacji", "ziarno", "łączenie wyników", "zapis w SAS"],
    "propozycja": [
        "Y_NEWDEPOSIT (jedyna zmienna z brakami: 2 879 obserwacji, 25,47%)",
        "monotoniczny (brakuje tylko Y)",
        "regresja logistyczna dla zmiennej skokowej (PROC MI, MONOTONE LOGISTIC)",
        ", ".join(wlaczone),
        "m = 100; efektywność (1 + γ/m)^(-1) sprawdzić po imputacji (slajd 154)",
        str(S.SEED),
        "model docelowy w każdym zbiorze, wyniki łączone regułami Rubina (PROC MIANALYZE)",
        "proc mi nimpute=100 seed=" + str(S.SEED) + "; class y contact house_loan poutcome "
        "education; monotone logistic(y = " + " ".join(efekty_sas) + "); var contact age "
        "house_loan poutcome education y;"]})
print("\nProjekt modelu imputacyjnego:\n", spec.to_string(index=False))
spec.to_csv(OUT / "s2_t12_projekt_modelu_imputacyjnego.csv", index=False)
print("\nPliki zapisane w:", OUT)
