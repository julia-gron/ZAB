# =============================================================================
# Punkt 5 i aspekt innowacyjny: diagnostyka imputacji i analiza wrażliwości
# Lokalizacja danych i folderu wyników jest zdefiniowana w 00_setup.py.
# Uruchamiać PO 03_modele.py: diagnostyka korzysta z tych samych imputacji
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
import sys
import importlib
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
from patsy import dmatrix

S = importlib.import_module("00_setup")  # ścieżki, ziarno, load_clean()
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
OUT = S.WYNIKI
SEED = S.SEED

print("Wersje: Python", sys.version.split()[0], "| pandas", pd.__version__,
      "| numpy", np.__version__, "| scipy", scipy.__version__,
      "| statsmodels", statsmodels.__version__, "| matplotlib", matplotlib.__version__)

# Wygląd wykresów taki sam jak w 01_struktura.py i 02_mechanizm.py
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
d = S.load_clean()
y = d["y"].to_numpy(dtype=float)
braki = (d["r"] == 1).to_numpy()
kanal = d["contact"].astype(str).to_numpy()            # "0", "1" (A), "2" (B)
isA, isB = kanal == "1", kanal == "2"

# Model docelowy (te same kolumny co w 03_modele.py); contact[T.1] = beta1 - beta2
X_DOC = dmatrix("contact + age + house_loan + poutcome", d, return_type="dataframe")
K_AB = list(X_DOC.columns).index("contact[T.1]")
# Model imputacyjny z punktu 2 (ten sam co w 03_modele.py) i jego warianty
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
    """Reguły Rubina (jak w 03_modele.py) -> słownik z oceną, błędem, przedziałem, p, FMI."""
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
    03_modele.py) i model docelowy w każdej. deltaA, deltaB: przesunięcie logitu szansy
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
# 1. MIARY EFEKTYWNOŚCI IMPUTACJI (z reguł Rubina w 03_modele.py)
# =============================================================================
print("\n### 1. Miary efektywności imputacji wielokrotnej (M z 03_modele.py)")
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
# Dane replikowane: w każdej imputacji z 03_modele.py, z tymi samymi wylosowanymi
# parametrami modelu imputacyjnego, losujemy Y także dla klientów z obserwowanym Y.
# PPC p = odsetek imputacji, w których beta1 - beta2 z danych replikowanych jest
# większe niż z danych uzupełnionych. Wartości bliskie 0,5: model odtwarza kontrast;
# bliskie 0 lub 1: model imputacyjny systematycznie zaniża lub zawyża kontrast.
print("\n### 2. Predykcyjna wartość p a posteriori (PPC) dla beta1 - beta2")
param = pd.read_csv(OUT / "s3_parametry_imputacji.csv", index_col=0)
imp = pd.read_csv(OUT / "s3_imputacje.csv", index_col=0)
oceny_s3 = pd.read_csv(OUT / "s3_oceny_imputacji.csv")
X_GL = dmatrix(MODELE_IMP["główny"], d, return_type="dataframe")
assert list(X_GL.columns) == list(param.columns), "Kolumny modelu imputacyjnego niezgodne z 03_modele.py"
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
# kontrola: ocena z danych uzupełnionych jest taka sama jak w 03_modele.py
y0 = y.copy()
y0[braki] = imp["imp_1"].to_numpy()
assert abs(kontrast(y0)[0] - ppc_gl["ocena"].iloc[0]) < 1e-8, "Imputacje niezgodne z 03_modele.py"


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
# 6. WSPÓLNA TABELA WYNIKÓW (wiersze dopisywane do tabeli z 03_modele.py w części 4)
# =============================================================================
print("\n### 6. Tabela wyników tej części")
t_wyn = pd.DataFrame(wyniki)
print(t_wyn.round(4).to_string(index=False))
t_wyn.to_csv(OUT / "s4_tabela_wynikow.csv", index=False)
print("\nPliki s4_* zapisane w:", OUT)
