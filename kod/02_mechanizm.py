# =============================================================================
# Punkt 2: mechanizm braków danych w Y
# Czy prawdopodobieństwo braku Y zależy od cech i czy istnieją cechy,
# za pomocą których imputacja może skorygować oszacowania?
# Lokalizacja danych i folderu wyników jest zdefiniowana w 00_setup.py
# Pliki wynikowe tej części mają przedrostek s2_ (sekcja 2 raportu).
# =============================================================================
import sys
import importlib
import warnings
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
pd.set_option("display.max_rows", 200)

print("Wersje: Python", sys.version.split()[0], "| pandas", pd.__version__,
      "| numpy", np.__version__, "| scipy", scipy.__version__,
      "| statsmodels", statsmodels.__version__, "| matplotlib", matplotlib.__version__)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})

# 0. PRZYGOTOWANIE
d = S.load_clean()
OUT = S.WYNIKI
rng = np.random.default_rng(S.SEED)                      # ziarno z 00_setup (bootstrap AUC)
CECHY_SK = ["contact", "house_loan", "education", "poutcome"]   # cechy skokowe
CECHY = ["contact", "age", "house_loan", "poutcome", "education"]
INNE = ["age", "house_loan", "poutcome", "education"]           # cechy poza CONTACT
NAZWA = {"contact": "CONTACT", "age": "AGE", "house_loan": "HOUSING_LOAN",
         "poutcome": "POUTCOME", "education": "EDUCATION"}
# Kwintyle wieku (granice z całej próby) - do tabel i do wariantu z AGE w przedziałach
d["age_q"] = pd.qcut(d["age"], 5)
d["age_q"] = d["age_q"].cat.rename_categories(
    [f"{int(np.floor(i.left)) + 1}-{int(i.right)}" for i in d["age_q"].cat.categories])
obs = d[d["r"] == 0].copy()                              # tylko obserwowane Y
SR = d["r"].mean()                                       # średni odsetek braków
print(f"n = {len(d)}, braki Y: {int(d.r.sum())} ({100 * SR:.2f}%)")


def pl(x, n=1):
    """Liczba z przecinkiem dziesiętnym (zapis polski)."""
    return f"{x:.{n}f}".replace(".", ",")


def save(name, rect=(0, 0, 1, 1)):
    plt.tight_layout(rect=rect)
    plt.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close()


def fit(formula, data):
    """Regresja logistyczna (MLE). Ostrzeżenia o zbieżności są wypisywane, nie ukrywane."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = smf.logit(formula, data=data).fit(disp=0, maxiter=200)
    for x in w:
        print(f"  [ostrzeżenie w {formula}]: {x.message}")
    if not m.mle_retvals["converged"]:
        print(f"  [UWAGA] brak zbieżności: {formula}")
    return m


def lr_test(pelny, zred):
    """Test ilorazu wiarygodności dla modeli zagnieżdżonych (te same obserwacje)."""
    assert pelny.nobs == zred.nobs
    lr = 2 * (pelny.llf - zred.llf)
    df = int(round(pelny.df_model - zred.df_model))
    return lr, df, stats.chi2.sf(lr, df)


def type3(formula_terms, y, data):
    """Testy LR dla całych cech (Type 3): model pełny wobec modelu bez danej cechy."""
    pelny = fit(f"{y} ~ " + " + ".join(formula_terms.values()), data)
    rows = []
    for nazwa, term in formula_terms.items():
        reszta = [t for k, t in formula_terms.items() if k != nazwa]
        zred = fit(f"{y} ~ " + (" + ".join(reszta) if reszta else "1"), data)
        lr, df, p = lr_test(pelny, zred)
        rows.append([nazwa, lr, df, p])
    return pelny, pd.DataFrame(rows, columns=["cecha", "LR chi2", "df", "p"]).set_index("cecha")


def or_table(m):
    """Ilorazy szans z 95% przedziałami Walda (bez wyrazu wolnego)."""
    ci = m.conf_int()
    t = pd.DataFrame({"beta": m.params, "SE": m.bse, "OR": np.exp(m.params),
                      "OR dolna 95%": np.exp(ci[0]), "OR górna 95%": np.exp(ci[1]),
                      "p (Wald)": m.pvalues})
    return t.drop(index="Intercept")


def auc_ci(y, p, B=1000):
    """AUC z 95% przedziałem bootstrap (percentylowym), ziarno z 00_setup."""
    y, p = np.asarray(y), np.asarray(p)
    a = roc_auc_score(y, p)
    boot = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boot.append(roc_auc_score(y[i], p[i]))
    return a, *np.percentile(boot, [2.5, 97.5])


# Kodowanie cech w formułach: poziomy referencyjne ustawione w 00_setup
# (contact = 2, house_loan = yes, poutcome = unknown, education = secondary).
TERMS = {c: c for c in CECHY}

# Zmienne 0/1 do jawnie zapisanych interakcji CONTACT x cecha (kanał B = poziom odniesienia).
# cA x cecha = różnica efektu cechy między kanałem A a B (to ta różnica ma znaczenie
# dla kontrastu A wobec B); c0 x cecha = różnica między kanałem 0 a B.
d["cA"] = (d["contact"] == "1").astype(float)
d["c0"] = (d["contact"] == "0").astype(float)
DUM = {"age": ["age"], "house_loan": ["hl_no"],
       "poutcome": ["po_failure", "po_success", "po_other"],
       "education": ["ed_primary", "ed_tertiary", "ed_unknown"]}
d["hl_no"] = (d["house_loan"] == "no").astype(float)
for lv in ["failure", "success", "other"]:
    d[f"po_{lv}"] = (d["poutcome"] == lv).astype(float)
for lv in ["primary", "tertiary", "unknown"]:
    d[f"ed_{lv}"] = (d["education"] == lv).astype(float)
obs = d[d["r"] == 0].copy()
# W kanale 0 tylko 23 klientów ma poutcome inne niż 'unknown' (wśród obserwowanych Y: 16,
# a poutcome = success tylko 2 osoby, obie Y = 0 -> separacja). Dlatego c0 x poutcome
# pomijamy w obu modelach; cA x poutcome (A wobec B) zostaje.
POMIN = {("c0", "poutcome")}


def inter_terms(cechy):
    """Lista składników interakcji kanał x cecha dla podanych cech."""
    return [f"{k}:{x}" for c in cechy for k in ["cA", "c0"]
            if (k, c) not in POMIN for x in DUM[c]]


def interakcje(y, data, model_bazowy, age="age"):
    """Test LR: model z interakcjami (łącznie i osobno dla każdej cechy) wobec modelu bez nich.
    `age` = postać wieku w modelu bazowym (liniowo albo splajn)."""
    baza = f"{y} ~ " + " + ".join(c if c != "age" else age for c in CECHY)
    rows = [["wszystkie łącznie", *lr_test(fit(baza + " + " + " + ".join(inter_terms(INNE)), data),
                                            model_bazowy)]]
    for c in INNE:
        rows.append([f"contact x {c}", *lr_test(fit(baza + " + " + " + ".join(inter_terms([c])), data),
                                                 model_bazowy)])
    return pd.DataFrame(rows, columns=["interakcja", "LR chi2", "df", "p"]).set_index("interakcja")

# =============================================================================
# 1. OPIS ZALEŻNOŚCI BRAKU OD CECH
# =============================================================================
print("\n### 1. Odsetek braków Y w kategoriach cech skokowych (test chi-kwadrat)")
rows, testy = [], []
for c in CECHY_SK + ["age_q"]:
    ct = pd.crosstab(d[c], d["r"])
    chi = stats.chi2_contingency(ct)
    # V Cramera jako miara siły związku (niezależna od liczności próby)
    v = np.sqrt(chi[0] / (ct.values.sum() * (min(ct.shape) - 1)))
    testy.append([c, chi[0], chi[2], chi[1], v])
    n, k = ct.sum(axis=1), ct[1]
    lo, hi = proportion_confint(k, n, method="wilson")
    for kat in ct.index:
        rows.append([c, str(kat), int(n[kat]), int(k[kat]), k[kat] / n[kat], lo[kat], hi[kat]])
t_r_kat = pd.DataFrame(rows, columns=["cecha", "kategoria", "n", "braki Y", "odsetek braków",
                                      "dolna 95%", "górna 95%"])
t_chi = pd.DataFrame(testy, columns=["cecha", "chi2", "df", "p", "V Cramera"]).set_index("cecha")
print(t_r_kat.round(4).to_string(index=False))
print("\nTesty niezależności r od cechy:\n", t_chi.round(4))
t_r_kat.round(4).to_csv(OUT / "s2_t1_braki_wg_kategorii.csv", index=False)
t_chi.round(4).to_csv(OUT / "s2_t2_chi2_braki.csv")

print("\n### 1b. AGE według r")
a0, a1 = d.loc[d.r == 0, "age"], d.loc[d.r == 1, "age"]
welch = stats.ttest_ind(a0, a1, equal_var=False)
mw = stats.mannwhitneyu(a0, a1)
ks = stats.ks_2samp(a0, a1)
sd_pooled = np.sqrt((a0.var() + a1.var()) / 2)
t_age = pd.DataFrame({
    "statystyka": ["średnia Y obserwowane", "średnia brak Y", "różnica średnich",
                   "SMD", "t Welcha", "p (t Welcha)", "p (Mann-Whitney)",
                   "D Kołmogorowa-Smirnowa", "p (K-S)"],
    "wartość": [a0.mean(), a1.mean(), a0.mean() - a1.mean(),
                (a0.mean() - a1.mean()) / sd_pooled, welch.statistic, welch.pvalue,
                mw.pvalue, ks.statistic, ks.pvalue]})
print(t_age.round(4).to_string(index=False))
t_age.round(4).to_csv(OUT / "s2_t3_age_wg_r.csv", index=False)

# Wykres zbiorczy: odsetek braków w kategoriach każdej cechy + linia średniego odsetka
tyt = {"contact": "Kanał (CONTACT)", "house_loan": "Kredyt mieszk.",
       "education": "Wykształcenie", "poutcome": "Poprz. kampania", "age_q": "Wiek (kwintyle)"}
krotkie = {"contact": {"0": "0", "1": "A", "2": "B"}}
fig, axs = plt.subplots(1, 5, figsize=(16, 4.4), sharey=True,
                        gridspec_kw={"width_ratios": [3, 2, 4, 4, 5]})
for ax, c in zip(axs, CECHY_SK + ["age_q"]):
    t = t_r_kat[t_r_kat.cecha == c]
    if c == "contact":                                   # kolejność 0, A, B
        t = t.set_index("kategoria").loc[["0", "1", "2"]].reset_index()
    lab = [krotkie.get(c, {}).get(k, k) for k in t.kategoria]
    est = 100 * t["odsetek braków"]
    err = [100 * (t["odsetek braków"] - t["dolna 95%"]), 100 * (t["górna 95%"] - t["odsetek braków"])]
    ax.bar(lab, est, yerr=err, capsize=4, color="#4c72b0", alpha=.85, error_kw={"elinewidth": 1.1})
    ax.axhline(100 * SR, ls="--", c="k", lw=1)
    for i, n in enumerate(t["n"]):
        ax.text(i, 1, f"n={n}", ha="center", fontsize=7, color="white", rotation=90, va="bottom")
    p = t_chi.loc[c, "p"]
    ax.set_title(f"{tyt[c]}\nchi2: p = {pl(p, 3) if p >= .001 else '< 0,001'}")
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.grid(True, alpha=.3)
axs[0].set_ylabel("% klientów bez wartości Y")
axs[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
fig.suptitle("Odsetek braków Y w kategoriach cech", fontweight="bold")
fig.text(.01, .01, f"Linia przerywana: średni odsetek braków w próbie ({pl(100 * SR)}%). "
         "Słupki błędów: 95% przedział Wilsona.", fontsize=8, color="0.3")
save("s2_w1_braki_wg_cech.png", rect=(0, .04, 1, .95))

# Rozkład wieku według r
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
bins = np.arange(15, 100, 3)
ax[0].hist(a0, bins=bins, density=True, alpha=.55, color="#1f77b4", label="Y obserwowane (r = 0)")
ax[0].hist(a1, bins=bins, density=True, alpha=.55, color="#d62728", label="brak Y (r = 1)")
ax[0].set(title="Rozkład wieku według r", xlabel="Wiek (lata)", ylabel="Gęstość")
ax[0].legend(frameon=False)
for x, col, lab in [(a0, "#1f77b4", "r = 0"), (a1, "#d62728", "r = 1")]:
    xs = np.sort(x)
    ax[1].step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=col, label=lab)
ax[1].set(title=f"Dystrybuanty empiryczne (K-S: p = {pl(ks.pvalue, 3)})",
          xlabel="Wiek (lata)", ylabel="F(wiek)")
ax[1].legend(frameon=False)
for a in ax:
    a.yaxis.grid(True, alpha=.3)
save("s2_w2_age_wg_r.png")

# =============================================================================
# 2. MODEL PRAWDOPODOBIEŃSTWA BRAKU: r ~ contact + age + house_loan + poutcome + education
# =============================================================================
print("\n### 2. Model logistyczny braku Y")
m_r, t3_r = type3(TERMS, "r", d)
print(m_r.summary2().tables[1].round(4))
or_r = or_table(m_r)
print("\nIlorazy szans braku Y:\n", or_r.round(4))
print("\nTesty LR dla całych cech (Type 3), model braku:\n", t3_r.round(4))
m_r0 = fit("r ~ 1", d)
lr_all = lr_test(m_r, m_r0)
auc_r = auc_ci(d.r, m_r.predict(d))
print(f"Łączny test LR wszystkich cech: chi2 = {lr_all[0]:.2f}, df = {lr_all[1]}, p = {lr_all[2]:.4f}")
print(f"AUC modelu braku = {auc_r[0]:.4f} (95% bootstrap: {auc_r[1]:.4f}-{auc_r[2]:.4f})")
print(f"Pseudo-R2 McFaddena = {m_r.prsquared:.5f}")
or_r.round(4).to_csv(OUT / "s2_t4_model_braku_or.csv")
t3_r.round(4).to_csv(OUT / "s2_t5_model_braku_type3.csv")

# 2a. Interakcje CONTACT x każda cecha: jeden łączny test LR i testy osobne
print("\n### 2a. Interakcje CONTACT x cecha w modelu braku")
t_int_r = interakcje("r", d, m_r)
print(t_int_r.round(4))
print("Uwaga: interakcja c0 x poutcome pominięta (zbyt mało obserwacji, patrz komentarz wyżej).")

# Kierunek interakcji z wiekiem: efekt AGE na szansę braku osobno w każdym kanale
rows = []
for k, nazwa in [("0", "0"), ("1", "A"), ("2", "B")]:
    m_k = fit("r ~ age", d[d.contact == k])
    ci = m_k.conf_int().loc["age"]
    rows.append([nazwa, int(m_k.nobs), np.exp(10 * m_k.params["age"]), np.exp(10 * ci[0]),
                 np.exp(10 * ci[1]), m_k.pvalues["age"]])
t_age_k = pd.DataFrame(rows, columns=["kanał", "n", "OR braku na 10 lat", "dolna 95%",
                                      "górna 95%", "p"]).set_index("kanał")
print("\nEfekt wieku na szansę braku Y w kanałach:\n", t_age_k.round(4))
t_age_k.round(4).to_csv(OUT / "s2_t6a_wiek_braki_w_kanalach.csv")

# 2b. Nieliniowość AGE: B-splajn (4 st. swobody) i AGE w kwintylach
print("\n### 2b. Nieliniowość AGE w modelu braku")
reszta = " + ".join(c for c in CECHY if c != "age")
m_r_bs = fit(f"r ~ bs(age, df=4) + {reszta}", d)
m_r_q = fit(f"r ~ age_q + {reszta}", d)
lr_bs = lr_test(m_r_bs, m_r)
t_nl_r = pd.DataFrame({"model": ["AGE liniowo", "AGE: B-splajn, df=4", "AGE w kwintylach"],
                       "logL": [m_r.llf, m_r_bs.llf, m_r_q.llf],
                       "AIC": [m_r.aic, m_r_bs.aic, m_r_q.aic]}).set_index("model")
print(t_nl_r.round(2))
print(f"LR splajn wobec liniowego: chi2 = {lr_bs[0]:.2f}, df = {lr_bs[1]}, p = {lr_bs[2]:.4f}")

# 2c. Test MCAR Little'a (uzupełnienie): EM dla wielowymiarowego rozkładu normalnego
print("\n### 2c. Test MCAR Little'a")


def little_mcar(X):
    """Test Little'a (1988). X: macierz liczbowa z NaN. Zwraca (d2, df, p)."""
    X = np.asarray(X, float)
    n, p = X.shape
    mu = np.nanmean(X, axis=0)
    sig = np.diag(np.nanvar(X, axis=0))
    M = np.isnan(X)
    for _ in range(500):                                 # algorytm EM
        Xh, C = X.copy(), np.zeros((p, p))
        for i in np.where(M.any(axis=1))[0]:
            m_, o_ = M[i], ~M[i]
            B = sig[np.ix_(m_, o_)] @ np.linalg.pinv(sig[np.ix_(o_, o_)])
            Xh[i, m_] = mu[m_] + B @ (X[i, o_] - mu[o_])
            C[np.ix_(m_, m_)] += sig[np.ix_(m_, m_)] - B @ sig[np.ix_(o_, m_)]
        mu_new = Xh.mean(axis=0)
        sig_new = (Xh - mu_new).T @ (Xh - mu_new) / n + C / n
        konwergencja = np.max(np.abs(mu_new - mu)) < 1e-8 and np.max(np.abs(sig_new - sig)) < 1e-8
        mu, sig = mu_new, sig_new
        if konwergencja:
            break
    d2, df = 0.0, 0
    wzorce = pd.DataFrame(M).value_counts().index
    for w in wzorce:
        w = np.array(w)
        idx = (M == w).all(axis=1)
        o_ = ~w
        diff = X[idx][:, o_].mean(axis=0) - mu[o_]
        d2 += idx.sum() * diff @ np.linalg.pinv(sig[np.ix_(o_, o_)]) @ diff
        df += o_.sum()
    df -= p
    return d2, df, stats.chi2.sf(d2, df)


Xl = pd.concat([d[["y", "age"]],
                pd.get_dummies(d[CECHY_SK], drop_first=True, dtype=float)], axis=1)
little = little_mcar(Xl)
print(f"Little: d2 = {little[0]:.2f}, df = {little[1]}, p = {little[2]:.4f}")
print("Test zakłada wielowymiarowy rozkład normalny (tu: zmienne 0/1), więc jest tylko uzupełnieniem.")

# 2d. Model braku z interakcją CONTACT x AGE (jedyna istotna interakcja w modelu braku)
m_r_ca = fit("r ~ " + " + ".join(CECHY) + " + cA:age + c0:age", d)
lr_r_ca = lr_test(m_r_ca, m_r)
auc_r_ca = auc_ci(d.r, m_r_ca.predict(d))
print(f"\nModel braku + CONTACT x AGE: LR = {lr_r_ca[0]:.2f}, df = {lr_r_ca[1]}, p = {lr_r_ca[2]:.4f}; "
      f"AIC {m_r.aic:.1f} -> {m_r_ca.aic:.1f}; AUC = {auc_r_ca[0]:.4f} "
      f"(95%: {auc_r_ca[1]:.4f}-{auc_r_ca[2]:.4f})")

# 2e. Przewidywane P(r = 0 | X) dla każdej obserwacji -> wagi IPW (Osoba 4).
# Wiersz = numer wiersza w dane_czyste.csv (liczony od 0). Zalecana kolumna: p_obs_int.
p_obs = pd.DataFrame({"wiersz": np.arange(len(d)), "r": d["r"],
                      "p_obs": 1 - m_r.predict(d),            # model z samymi efektami głównymi
                      "p_obs_int": 1 - m_r_ca.predict(d)})    # + CONTACT x AGE (zalecany)
p_obs.to_csv(OUT / "s2_p_obserwacji.csv", index=False)
print("P(r = 0 | X) zapisane do s2_p_obserwacji.csv. Zakres p_obs_int:",
      f"{p_obs.p_obs_int.min():.4f}-{p_obs.p_obs_int.max():.4f}")

# Zbiorcza tabela testów modelu braku
t_mb = pd.concat([
    t3_r.rename(index=lambda c: f"Type 3: {c}"),
    pd.DataFrame([lr_all], columns=["LR chi2", "df", "p"], index=["wszystkie cechy łącznie"]),
    t_int_r.rename(index=lambda c: f"interakcje: {c}"),
    pd.DataFrame([lr_bs], columns=["LR chi2", "df", "p"], index=["nieliniowość AGE (B-splajn)"]),
    pd.DataFrame([lr_r_ca], columns=["LR chi2", "df", "p"], index=["CONTACT x AGE (sama)"]),
    pd.DataFrame([little], columns=["LR chi2", "df", "p"], index=["test MCAR Little'a (d2)"])])
t_mb.round(4).to_csv(OUT / "s2_t6_model_braku_testy.csv")

# =============================================================================
# 3. ZWIĄZEK CECH Z Y (tylko obserwowane Y)
# =============================================================================
print("\n### 3. Model logistyczny Y na obserwowanych Y (wszystkie cechy)")
m_y, t3_y = type3(TERMS, "y", obs)
or_y = or_table(m_y)
print("Ilorazy szans Y = 1:\n", or_y.round(4))
print("\nTesty LR dla całych cech (Type 3), model Y:\n", t3_y.round(4))
auc_y = auc_ci(obs.y, m_y.predict(obs))
print(f"AUC modelu Y = {auc_y[0]:.4f} (95% bootstrap: {auc_y[1]:.4f}-{auc_y[2]:.4f})")
or_y.round(4).to_csv(OUT / "s2_t7_model_y_or.csv")

# 3a. Czy EDUCATION wnosi informację o Y ponad cechy modelu docelowego?
m_doc = fit("y ~ age + poutcome + house_loan + contact", obs)        # model docelowy (CC)
lr_edu = lr_test(m_y, m_doc)
auc_doc = roc_auc_score(obs.y, m_doc.predict(obs))
print(f"\nEDUCATION ponad model docelowy: LR chi2 = {lr_edu[0]:.2f}, df = {lr_edu[1]}, "
      f"p = {lr_edu[2]:.4f}; AUC {auc_doc:.4f} -> {auc_y[0]:.4f}")
b_doc = m_doc.params["contact[T.1]"]
b_pel = m_y.params["contact[T.1]"]
print(f"beta1 - beta2 (A wobec B): model docelowy {b_doc:.4f} (OR {np.exp(b_doc):.3f}), "
      f"z EDUCATION {b_pel:.4f} (OR {np.exp(b_pel):.3f})")

# 3b. Nieliniowość AGE w modelu Y
print("\n### 3b. Nieliniowość AGE w modelu Y")
AGE_BS = "bs(age, df=4)"
m_y_bs = fit(f"y ~ {AGE_BS} + {reszta}", obs)
lr_bs_y = lr_test(m_y_bs, m_y)
print(f"LR splajn wobec liniowego: chi2 = {lr_bs_y[0]:.2f}, df = {lr_bs_y[1]}, p = {lr_bs_y[2]:.4f}; "
      f"AIC {m_y.aic:.1f} -> {m_y_bs.aic:.1f}")

# 3c. Interakcje CONTACT x cecha w modelu Y, przy AGE liniowym i przy AGE jako splajnie.
# Przy liniowym AGE interakcja z wiekiem może być pozorna: zależność Y od wieku ma kształt U,
# a klienci kanału B są starsi, więc liniowe nachylenia różnią się między kanałami.
print("\n### 3c. Interakcje CONTACT x cecha w modelu Y")
t_int_y_lin = interakcje("y", obs, m_y)
t_int_y = interakcje("y", obs, m_y_bs, age=AGE_BS)
print("AGE liniowo:\n", t_int_y_lin.round(4))
print("AGE jako B-splajn (te testy idą do tabeli decyzyjnej):\n", t_int_y.round(4))

# Efekt wieku na szansę zakupu w kanałach (para do tabeli s2_t6a)
rows = []
for k, nazwa in [("0", "0"), ("1", "A"), ("2", "B")]:
    m_k = fit("y ~ age", obs[obs.contact == k])
    ci = m_k.conf_int().loc["age"]
    rows.append([nazwa, int(m_k.nobs), np.exp(10 * m_k.params["age"]), np.exp(10 * ci[0]),
                 np.exp(10 * ci[1]), m_k.pvalues["age"]])
t_age_ky = pd.DataFrame(rows, columns=["kanał", "n", "OR zakupu na 10 lat", "dolna 95%",
                                       "górna 95%", "p"]).set_index("kanał")
print("\nEfekt wieku na szansę zakupu (obserwowane Y) w kanałach:\n", t_age_ky.round(4))
t_age_ky.round(4).to_csv(OUT / "s2_t7a_wiek_y_w_kanalach.csv")

t_my = pd.concat([
    t3_y.rename(index=lambda c: f"Type 3: {c}"),
    pd.DataFrame([lr_edu], columns=["LR chi2", "df", "p"], index=["EDUCATION ponad model docelowy"]),
    pd.DataFrame([lr_bs_y], columns=["LR chi2", "df", "p"], index=["nieliniowość AGE (B-splajn)"]),
    t_int_y_lin.rename(index=lambda c: f"interakcje (AGE liniowo): {c}"),
    t_int_y.rename(index=lambda c: f"interakcje (AGE splajn): {c}")])
t_my.round(4).to_csv(OUT / "s2_t8_model_y_testy.csv")

# Wykres: wiek, brak Y i zakup w kanałach (sedno mechanizmu)
PRZ = [17, 30, 40, 50, 60, 100]
ETP = ["18-30", "31-40", "41-50", "51-60", "61+"]
d["age_p"] = pd.cut(d["age"], PRZ, labels=ETP)
obs["age_p"] = pd.cut(obs["age"], PRZ, labels=ETP)
KOL = {"0": ("#8c8c8c", "0: brak komunikacji"), "1": ("#1f77b4", "A: kanał A"),
       "2": ("#ff7f0e", "B: kanał B")}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
rows = []
for ax, (dane, kol, tytul) in zip(axs, [(d, "r", "Odsetek braków Y"),
                                        (obs, "y", "Odsetek zakupów (obserwowane Y)")]):
    for k, (c, lab) in KOL.items():
        g = dane[dane.contact == k].groupby("age_p", observed=False)[kol].agg(["sum", "count"])
        g = g[g["count"] >= 20]                          # pomijamy przedziały z n < 20
        lo, hi = proportion_confint(g["sum"], g["count"], method="wilson")
        x = np.array([ETP.index(i) for i in g.index]) + {"0": -.12, "1": 0, "2": .12}[k]
        p = g["sum"] / g["count"]
        ax.errorbar(x, 100 * p, yerr=[100 * (p - lo), 100 * (hi - p)], fmt="o-", color=c,
                    capsize=3, label=lab)
        for i, row in g.iterrows():
            rows.append([tytul, lab, i, int(row["count"]), row["sum"] / row["count"]])
    ax.set_xticks(range(len(ETP)))
    ax.set_xticklabels(ETP)
    ax.set(title=tytul + " według wieku i kanału", xlabel="Wiek (lata)", ylabel="%")
    ax.yaxis.grid(True, alpha=.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
axs[0].axhline(100 * SR, ls="--", c="k", lw=1)
axs[0].legend(frameon=False, fontsize=8)
fig.text(.01, .01, "Słupki błędów: 95% przedział Wilsona. Przedziały wieku z n < 20 pominięte "
         "(kanał 0, wiek 61+: n = 4). Linia przerywana: średni odsetek braków.",
         fontsize=8, color="0.3")
save("s2_w4_wiek_kanal_braki_zakup.png", rect=(0, .04, 1, 1))
pd.DataFrame(rows, columns=["wykres", "kanał", "wiek", "n", "odsetek"]).round(4).to_csv(
    OUT / "s2_t8a_wiek_kanal_odsetki.csv", index=False)

# Wykres: ilorazy szans w modelu braku i w modelu Y obok siebie
etyk = {"contact[T.0]": "CONTACT 0 wobec B", "contact[T.1]": "CONTACT A wobec B",
        "house_loan[T.no]": "brak kredytu (wobec yes)",
        "poutcome[T.failure]": "poutcome failure", "poutcome[T.success]": "poutcome success",
        "poutcome[T.other]": "poutcome other", "education[T.primary]": "edukacja primary",
        "education[T.tertiary]": "edukacja tertiary", "education[T.unknown]": "edukacja unknown",
        "age": "wiek (+10 lat)"}


def or10(t):
    """OR dla wieku przeliczony na 10 lat, żeby był czytelny na wykresie."""
    t = t.copy()
    t.loc["age", ["OR", "OR dolna 95%", "OR górna 95%"]] **= 10
    return t


# Wspólna skala osi X: od razu widać, że efekty w modelu braku są znikome wobec modelu Y
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, sharex=True)
for ax, (t, tytul, kol) in zip(axs, [(or10(or_r), "Model braku: P(r = 1)", "#d62728"),
                                     (or10(or_y), "Model Y: P(Y = 1), obserwowane Y", "#1f77b4")]):
    poz = np.arange(len(t))[::-1]
    ax.errorbar(t["OR"], poz, xerr=[t["OR"] - t["OR dolna 95%"], t["OR górna 95%"] - t["OR"]],
                fmt="o", color=kol, capsize=3)
    ax.axvline(1, ls="--", c="k", lw=1)
    ax.set_xscale("log")
    ax.set_xticks([.25, .5, 1, 2, 4, 8, 16])
    ax.minorticks_off()
    ax.set_title(tytul)
    ax.set_xlabel("Iloraz szans (skala log), 95% PU")
    ax.xaxis.grid(True, alpha=.3, which="both")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
axs[0].set_yticks(np.arange(len(or_r))[::-1])
axs[0].set_yticklabels([etyk.get(i, i) for i in or_r.index])
fig.suptitle("Te same cechy w modelu braku Y i w modelu Y", fontweight="bold")
save("s2_w3_or_braku_i_y.png", rect=(0, 0, 1, .95))


# =============================================================================
# 4. TABELA DECYZYJNA I PROJEKT MODELU IMPUTACYJNEGO
# =============================================================================
print("\n### 4a. Porównanie kandydatów na model imputacyjny Y (obserwowane Y)")
# Wszystkie warianty zawierają cechy modelu docelowego oraz EDUCATION (cecha pomocnicza).
GL = "contact + house_loan + poutcome + education"
INT_AGE, INT_HL = "cA:age + c0:age", "cA:hl_no + c0:hl_no"
INT_RESZTA = " + ".join(inter_terms(["poutcome", "education"]))
KAND = {
    "K1: efekty główne, AGE liniowo": f"y ~ {GL} + age",
    "K2: efekty główne, AGE splajn": f"y ~ {GL} + {AGE_BS}",
    "K3: K2 + CONTACT x AGE": f"y ~ {GL} + {AGE_BS} + {INT_AGE}",
    "K4: K2 + CONTACT x HOUSING_LOAN": f"y ~ {GL} + {AGE_BS} + {INT_HL}",
    "K5: K2 + CONTACT x (AGE, HOUSING_LOAN)": f"y ~ {GL} + {AGE_BS} + {INT_AGE} + {INT_HL}",
    "K6: K2 + CONTACT x splajn AGE": f"y ~ {GL} + {AGE_BS} + cA:{AGE_BS} + c0:{AGE_BS}",
    "K7: K5 + CONTACT x (POUTCOME, EDUCATION)": f"y ~ {GL} + {AGE_BS} + {INT_AGE} + {INT_HL} + {INT_RESZTA}",
}
rows, modele = [], {}
for nazwa, f in KAND.items():
    m = fit(f, obs)
    modele[nazwa] = m
    rows.append([nazwa, int(m.df_model) + 1, m.llf, m.aic, m.bic,
                 roc_auc_score(obs.y, m.predict(obs))])
t_kand = pd.DataFrame(rows, columns=["model", "liczba parametrów", "logL", "AIC", "BIC", "AUC"]
                      ).set_index("model")
t_kand["delta AIC"] = t_kand["AIC"] - t_kand["AIC"].min()
WYBOR = t_kand["AIC"].idxmin()
print(t_kand.round(3))
print("Najniższe AIC:", WYBOR, "| najniższe BIC:", t_kand["BIC"].idxmin())
lr_k5_k2 = lr_test(modele["K5: K2 + CONTACT x (AGE, HOUSING_LOAN)"], modele["K2: efekty główne, AGE splajn"])
print(f"LR K5 wobec K2: chi2 = {lr_k5_k2[0]:.2f}, df = {lr_k5_k2[1]}, p = {lr_k5_k2[2]:.4f}")
t_kand.round(4).to_csv(OUT / "s2_t9_kandydaci_modelu_imputacyjnego.csv")

print("\n### 4b. Tabela decyzyjna")
ALFA = .05


def rola(z_r, z_y):
    """Reguła z metodyki: związek z r i z Y usuwa obciążenie; tylko z Y poprawia precyzję;
    tylko z r niewiele daje; bez związku - można pominąć."""
    if z_r and z_y:
        return "usuwa obciążenie"
    if z_y:
        return "poprawia precyzję"
    if z_r:
        return "niewiele daje"
    return "bez wpływu"


# (element, czy w modelu docelowym, test dla r, test dla Y)
ELEM = [
    ("CONTACT", "tak", t3_r.loc["contact"], t3_y.loc["contact"]),
    ("AGE (liniowo)", "tak", t3_r.loc["age"], t3_y.loc["age"]),
    ("AGE (nieliniowo, B-splajn)", "nie", pd.Series(lr_bs, index=["LR chi2", "df", "p"]),
     pd.Series(lr_bs_y, index=["LR chi2", "df", "p"])),
    ("HOUSING_LOAN", "tak", t3_r.loc["house_loan"], t3_y.loc["house_loan"]),
    ("POUTCOME", "tak", t3_r.loc["poutcome"], t3_y.loc["poutcome"]),
    ("EDUCATION", "nie", t3_r.loc["education"], t3_y.loc["education"]),
] + [(f"CONTACT x {NAZWA[c]}", "nie", t_int_r.loc[f"contact x {c}"], t_int_y.loc[f"contact x {c}"])
     for c in INNE]
# Decyzja: cechy modelu docelowego zawsze (zgodność modelu imputacyjnego z docelowym);
# pozostałe, gdy reguła daje "usuwa obciążenie" lub "poprawia precyzję". Wyjątek (strategia
# inkluzywna, Collins, Schafer i Kam 2001): element wyraźnie związany z r i słabo z Y
# (p(Y) < 0,10) też wchodzi, bo obciążenie zależy od iloczynu obu związków, a zbędny
# predyktor kosztuje tylko trochę precyzji.
ALFA_INKL = .10
rows = []
for nazwa, w_doc, tr, ty in ELEM:
    z_r, z_y = tr["p"] < ALFA, ty["p"] < ALFA
    rl = rola(z_r, z_y)
    if w_doc == "tak":
        decyzja = "włączyć (cecha modelu docelowego)"
    elif rl in ("usuwa obciążenie", "poprawia precyzję"):
        decyzja = "włączyć"
    elif z_r and ty["p"] < ALFA_INKL:
        decyzja = "włączyć (silny związek z r, słaby z Y)"
    else:
        decyzja = "pominąć (wariant w analizie wrażliwości)"
    rows.append([nazwa, w_doc, tr["LR chi2"], int(tr["df"]), tr["p"], ty["LR chi2"], int(ty["df"]),
                 ty["p"], "tak" if z_r else "nie", "tak" if z_y else "nie", rl, decyzja])
t_dec = pd.DataFrame(rows, columns=["element", "w modelu docelowym", "LR (r)", "df (r)", "p (r)",
                                    "LR (Y)", "df (Y)", "p (Y)", "związek z r", "związek z Y",
                                    "rola wg reguły", "decyzja"])
print(t_dec.round(4).to_string(index=False))
print("Testy dla r: model braku z AGE liniowym (splajn nieistotny); testy dla Y: model Y na"
      " obserwowanych Y, interakcje przy AGE jako splajnie.")
t_dec.round(4).to_csv(OUT / "s2_t10_tabela_decyzyjna.csv", index=False)

print("\n### 4c. Projekt modelu imputacyjnego (propozycja na bramkę)")
spec = pd.DataFrame({
    "element": ["zmienna imputowana", "metoda", "predyktory (efekty główne)", "postać AGE",
                "interakcje", "wyłączone z modelu", "liczba imputacji", "ziarno",
                "łączenie wyników", "warianty do analizy wrażliwości"],
    "propozycja": [
        "Y_NEWDEPOSIT; jedyna zmienna z brakami (2 879 obs., 25,47%)",
        "regresja logistyczna z losowaniem parametrów z rozkładu a posteriori (mice: 'logreg'); "
        "braki tylko w Y, więc wystarczy jeden krok bez iteracji FCS",
        "CONTACT, HOUSING_LOAN, POUTCOME, EDUCATION (pomocnicza)",
        "B-splajn sześcienny, 4 stopnie swobody (bs(age, df=4))",
        "CONTACT x AGE (liniowo), CONTACT x HOUSING_LOAN",
        "r (wskaźnik braku); CONTACT x POUTCOME i CONTACT x EDUCATION (nieistotne)",
        "pilotaż m = 20; docelowo m >= 100 x FMI kontrastu beta1 - beta2 (reguła von Hippela)",
        str(S.SEED),
        "reguły Rubina dla beta1 - beta2 i jego błędu, exp dopiero na końcu",
        "K2 (bez interakcji), K6 (CONTACT x splajn AGE), K7 (wszystkie interakcje); MID"]})
print(spec.to_string(index=False))
print("Formuła (patsy/statsmodels):", KAND["K5: K2 + CONTACT x (AGE, HOUSING_LOAN)"])
print("W R (mice): y ~ contact + house_loan + poutcome + education + splines::bs(age, df = 4)"
      " + contact:age + contact:house_loan")
spec.to_csv(OUT / "s2_t11_projekt_modelu_imputacyjnego.csv", index=False)

# Podsumowanie liczbowe do tekstu raportu
print("\n### Podsumowanie")
print(f"Model braku (efekty główne): AUC = {auc_r[0]:.3f}, łączny LR p = {lr_all[2]:.4f}, "
      f"Little p = {little[2]:.4f}")
print(f"Model braku + CONTACT x AGE: p = {lr_r_ca[2]:.4f}, AUC = {auc_r_ca[0]:.3f}")
print(f"Model Y: AUC = {auc_y[0]:.3f}, EDUCATION p = {lr_edu[2]:.4f}, splajn AGE p = {lr_bs_y[2]:.4g}, "
      f"interakcje (przy splajnie) p = {t_int_y.loc['wszystkie łącznie', 'p']:.4f}")
print("Wybrany model imputacyjny (min AIC):", WYBOR)
print("\nPliki zapisane w:", OUT)
