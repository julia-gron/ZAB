# =============================================================================
# Punkt 5 i aspekt innowacyjny: diagnostyka imputacji i analiza wrażliwości
# Lokalizacja danych i folderu wyników jest zdefiniowana w 00_setup.py.
# Pliki wynikowe tej części mają przedrostek s4_.
#
# Zawartość:
#   0. Narzędzia: regresja logistyczna (IRLS), macierze cech, imputacja, reguły Rubina
#   1. Imputacja wielokrotna przy MAR (model z punktu 2) + dane replikowane
#   2. Punkt 5: predykcyjna wartość p a posteriori (PPC) dla beta1 - beta2
#   3. Innowacja 1: analiza wrażliwości MNAR (delta-adjustment), punkt przełamania
#   4. Innowacja 3: IPW (ważenie odwrotnością prawdopodobieństwa obserwacji)
#   5. Innowacja 4: porównanie modeli imputacyjnych (w tym CONTACT x AGE)
#   6. Innowacja 5: efekt na skali prawdopodobieństwa (średni efekt krańcowy)
#   7. Innowacja 6: test na sztucznych brakach (znana prawda)
#   8. Wspólna tabela wyników
#
# Regresja logistyczna jest liczona tu bezpośrednio w numpy (algorytm IRLS, ten sam
# co w PROC LOGISTIC i statsmodels), bo imputacja jest powtarzana kilka tysięcy razy.
# Zgodność z 03_modele.py jest sprawdzana w kodzie (punkt 1).
# Model imputacyjny jest ten sam co w 03_modele.py: contact + age + house_loan +
# poutcome + education + contact:house_loan, regresja logistyczna z losowaniem
# parametrów z ich rozkładu (przybliżona imputacja bayesowska).
# =============================================================================
import sys
import time
import importlib
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use("Agg")                    # zapis wykresów do plików, bez okna
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from scipy.special import expit

S = importlib.import_module("00_setup")  # ścieżki, ziarno, load_clean()
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
OUT = S.WYNIKI
SEED = S.SEED
T0 = time.time()

print("Wersje: Python", sys.version.split()[0], "| pandas", pd.__version__,
      "| numpy", np.__version__, "| scipy", scipy.__version__,
      "| matplotlib", matplotlib.__version__)

# Wygląd wykresów taki sam jak w 01_struktura.py i 02_mechanizm.py
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})


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


# Liczby imputacji: pełne M jak w 03_modele.py; mniejsze M na siatce MNAR i w
# symulacji (tam liczy się położenie wyniku, a nie trzecie miejsce po przecinku).
M_PELNE = 500        # model główny, warianty i scenariusze do tabeli (mały błąd Monte Carlo)
M_SIATKA = 20        # siatka 9 x 9 do mapy ciepła
M_LINIE = 100        # gęste linie do wyznaczenia punktu przełamania
M_SYM = 10
R_SYM = 200          # liczba powtórzeń w teście na sztucznych brakach

# =============================================================================
# 0. NARZĘDZIA
# =============================================================================
d = S.load_clean()
n = len(d)
y = d["y"].to_numpy(dtype=float)
braki = d["r"].to_numpy() == 1          # r = 1: brak Y
kanal = d["contact"].astype(str).to_numpy()   # "0", "1" (A), "2" (B)
isA, isB, is0 = kanal == "1", kanal == "2", kanal == "0"
print(f"n = {n}, braki Y: {braki.sum()} ({100 * braki.mean():.2f}%)")


def dummies(s, poziomy):
    """Zmienne 0/1 dla poziomów `poziomy` (poziom referencyjny pomijamy)."""
    return np.column_stack([(s == p).to_numpy(dtype=float) for p in poziomy])


# Cechy modelu docelowego. Kolejność kolumn jest stała:
# [wyraz wolny, contact=0, contact=1 (A), age, house_loan=no, poutcome: failure, success, other]
# Poziom referencyjny contact = 2 (kanał B), więc współczynnik w kolumnie 2 to beta1 - beta2.
C0 = is0.astype(float)
CA = isA.astype(float)
AGE = d["age"].to_numpy(dtype=float)
HL_NO = (d["house_loan"] == "no").to_numpy(dtype=float)
POUT = dummies(d["poutcome"], ["failure", "success", "other"])
EDU = dummies(d["education"], ["primary", "tertiary", "unknown"])
X_DOC = np.column_stack([np.ones(n), C0, CA, AGE, HL_NO, POUT])
NAZWY_DOC = ["wyraz wolny", "contact 0 wobec B", "contact A wobec B", "age",
             "house_loan no", "poutcome failure", "poutcome success", "poutcome other"]
K_AB = 2                                  # indeks kolumny kontrastu A wobec B


def x_imp(wariant="glowny", kontakt=None):
    """Macierz cech modelu imputacyjnego.
    glowny       : model z punktu 2 (Osoba 2) i 03_modele.py:
                   contact + age + house_loan + poutcome + education + contact:house_loan
    bez_interakcji: bez contact:house_loan
    z_wiekiem    : glowny + contact:age (interakcja istotna w modelu braku, p = 0,001)
    bez_kanalu   : celowo błędny model bez CONTACT (kontrola czułości PPC)
    `kontakt` pozwala podstawić inne kolumny kanału (nieużywane w imputacji)."""
    c0, ca = (C0, CA) if kontakt is None else kontakt
    wiek_c = (AGE - AGE.mean()) / 10                 # wiek wyśrodkowany, w dekadach
    kol = [np.ones(n), c0, ca, AGE, HL_NO, POUT, EDU]
    if wariant == "bez_kanalu":
        kol = [np.ones(n), AGE, HL_NO, POUT, EDU]
    if wariant in ("glowny", "z_wiekiem"):
        kol += [c0 * HL_NO, ca * HL_NO]
    if wariant == "z_wiekiem":
        kol += [c0 * wiek_c, ca * wiek_c]
    return np.column_stack(kol)


def logit_fit(X, yv, w=None, robust=False, maxiter=50, tol=1e-10):
    """Regresja logistyczna metodą IRLS (Newtona-Raphsona).
    w      : wagi obserwacji (IPW); None = wszystkie równe 1
    robust : odporna macierz kowariancji (kanapkowa, HC0), potrzebna przy wagach
    Zwraca (beta, macierz kowariancji beta)."""
    w = np.ones(len(yv)) if w is None else w
    b = np.zeros(X.shape[1])
    for _ in range(maxiter):
        p = expit(X @ b)
        H = X.T @ (X * (w * p * (1 - p))[:, None])
        krok = np.linalg.solve(H, X.T @ (w * (yv - p)))
        b = b + krok
        if np.max(np.abs(krok)) < tol:
            break
    else:
        raise RuntimeError("Brak zbieżności IRLS")
    p = expit(X @ b)
    H = X.T @ (X * (w * p * (1 - p))[:, None])
    Hinv = np.linalg.inv(H)
    if not robust:
        return b, Hinv
    U = X * (w * (yv - p))[:, None]                  # wkłady do funkcji wynikowej
    return b, Hinv @ (U.T @ U) @ Hinv


def kontrast(yv, rows=None):
    """Model docelowy na danych (yv) -> (beta1 - beta2, wariancja, pełne beta, kowariancja)."""
    X = X_DOC if rows is None else X_DOC[rows]
    b, V = logit_fit(X, yv)
    return b[K_AB], V[K_AB, K_AB], b, V


def ame_ab(b, V):
    """Średni efekt krańcowy kanału A wobec B na skali prawdopodobieństwa:
    średnia po wszystkich klientach z P(Y=1 | kanał A) - P(Y=1 | kanał B), przy ich
    własnych pozostałych cechach. Wariancja z metody delta."""
    XA, XB = X_DOC.copy(), X_DOC.copy()
    XA[:, 1], XA[:, 2] = 0, 1                        # każdy klient "w kanale A"
    XB[:, 1], XB[:, 2] = 0, 0                        # każdy klient "w kanale B"
    pA, pB = expit(XA @ b), expit(XB @ b)
    g = (XA * (pA * (1 - pA))[:, None]).mean(0) - (XB * (pB * (1 - pB))[:, None]).mean(0)
    return (pA - pB).mean(), g @ V @ g


def rubin(oceny, wariancje):
    """Reguły Rubina (jak w 03_modele.py) -> słownik z oceną, błędem, df, p, FMI, RIV."""
    oceny, wariancje = np.asarray(oceny), np.asarray(wariancje)
    m = len(oceny)
    Q, W, B = oceny.mean(), wariancje.mean(), oceny.var(ddof=1)
    T = W + (1 + 1 / m) * B
    riv = (1 + 1 / m) * B / W                         # względny wzrost wariancji
    df = (m - 1) * (1 + 1 / riv) ** 2
    fmi = (riv + 2 / (df + 3)) / (riv + 1)
    se = np.sqrt(T)
    tkr = stats.t.ppf(.975, df)
    return {"ocena": Q, "błąd": se, "dolna": Q - tkr * se, "górna": Q + tkr * se,
            "p": 2 * stats.t.sf(abs(Q / se), df), "df": df, "FMI": fmi, "RIV": riv, "m": m}


def wiersz(metoda, r, fmi=True):
    """Wiersz wspólnej tabeli wyników (format jak s3_tabela_wynikow.csv)."""
    return {"metoda": metoda, "beta1 - beta2": r["ocena"], "błąd": r["błąd"],
            "OR A wobec B": np.exp(r["ocena"]), "OR dolna 95%": np.exp(r["dolna"]),
            "OR górna 95%": np.exp(r["górna"]), "p": r["p"],
            "FMI": r.get("FMI", np.nan) if fmi else np.nan}


class Imputator:
    """Imputacja Y regresją logistyczną z losowaniem parametrów (jak `imputuj()` w
    03_modele.py), rozszerzona o dwie rzeczy potrzebne w diagnostyce:
      - replikacja: losowanie Y także dla klientów z obserwowanym Y (PPC),
      - przesunięcie delta na skali logitu dla brakujących Y, osobno w kanale A i B
        (MNAR; delta = 0 oznacza MAR; w kanale 0 delta = 0)."""

    def __init__(self, wariant="glowny"):
        self.X = x_imp(wariant)
        self.b, self.V = logit_fit(self.X[~braki], y[~braki])   # dopasowanie na obserwowanych Y

    def losuj(self, rng, deltaA=0.0, deltaB=0.0, replikuj=False):
        beta = rng.multivariate_normal(self.b, self.V)        # niepewność parametrów
        eta = self.X @ beta
        y_uzup = y.copy()
        eta_mis = eta[braki] + deltaA * isA[braki] + deltaB * isB[braki]
        y_uzup[braki] = rng.binomial(1, expit(eta_mis))
        if not replikuj:
            return y_uzup
        # dane replikowane: obserwowane Y zastąpione losowaniem z tego samego modelu
        # (te same parametry beta), imputowane Y bez zmian (jak argument where w mice)
        y_rep = y_uzup.copy()
        y_rep[~braki] = rng.binomial(1, expit(eta[~braki]))
        return y_uzup, y_rep


def mi(imp, M, seed, deltaA=0.0, deltaB=0.0, replikuj=False, z_ame=False):
    """M imputacji + model docelowy w każdej. Zwraca DataFrame z wynikiem każdej imputacji."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(M):
        out = imp.losuj(rng, deltaA, deltaB, replikuj)
        y_uzup, y_rep = out if replikuj else (out, None)
        est, var, b, V = kontrast(y_uzup)
        w = {"imputacja": i + 1, "ocena": est, "wariancja": var,
             "Y=1 imputowane A": y_uzup[braki & isA].mean(),
             "Y=1 imputowane B": y_uzup[braki & isB].mean(),
             "Y=1 imputowane 0": y_uzup[braki & is0].mean()}
        if z_ame:
            w["AME"], w["AME wariancja"] = ame_ab(b, V)
        if replikuj:
            w["ocena replikowana"] = kontrast(y_rep)[0]
            for nazwa, maska in [("A", isA), ("B", isB), ("0", is0)]:
                w[f"Y=1 uzupełnione {nazwa}"] = y_uzup[maska].mean()
                w[f"Y=1 replikowane {nazwa}"] = y_rep[maska].mean()
        rows.append(w)
    return pd.DataFrame(rows)


wyniki = []                                    # wiersze wspólnej tabeli wyników

# =============================================================================
# 1. IMPUTACJA PRZY MAR (model z punktu 2) + DANE REPLIKOWANE
# =============================================================================
print("\n### 1. Imputacja wielokrotna przy MAR, model z punktu 2, M =", M_PELNE)
est_cc, var_cc, b_cc, V_cc = kontrast(y[~braki], rows=~braki)
s3 = pd.read_csv(OUT / "s3_tabela_wynikow.csv")
print(f"Kompletne przypadki: beta1 - beta2 = {est_cc:.4f}, SE = {np.sqrt(var_cc):.4f} "
      f"(03_modele.py: {s3.iloc[0]['beta1 - beta2']:.4f}, SE = {s3.iloc[0]['błąd']:.4f})")
assert abs(est_cc - s3.iloc[0]["beta1 - beta2"]) < 1e-6, "Niezgodność z 03_modele.py"
r_cc = {"ocena": est_cc, "błąd": np.sqrt(var_cc),
        "dolna": est_cc - 1.959964 * np.sqrt(var_cc), "górna": est_cc + 1.959964 * np.sqrt(var_cc),
        "p": 2 * stats.norm.sf(abs(est_cc) / np.sqrt(var_cc))}

imp_glowny = Imputator("glowny")
mi_glowny = mi(imp_glowny, M_PELNE, SEED, replikuj=True, z_ame=True)
r_mar = rubin(mi_glowny["ocena"], mi_glowny["wariancja"])
print(f"MI (MAR): beta1 - beta2 = {r_mar['ocena']:.4f}, SE = {r_mar['błąd']:.4f}, "
      f"FMI = {r_mar['FMI']:.3f} (03_modele.py: {s3.iloc[1]['beta1 - beta2']:.4f}, "
      f"SE = {s3.iloc[1]['błąd']:.4f}, FMI = {s3.iloc[1]['FMI']:.3f})")
# Ocena różni się od 03_modele.py tylko losowością imputacji: inny strumień liczb losowych.
# Błąd Monte Carlo średniej z M imputacji = sqrt(B / M).
mc = np.sqrt(mi_glowny["ocena"].var(ddof=1) / M_PELNE)
roznica = r_mar["ocena"] - s3.iloc[1]["beta1 - beta2"]
print(f"Różnica względem 03_modele.py: {roznica:.4f} (błąd Monte Carlo ok. {mc:.4f} na każdy przebieg)")
mi_glowny.to_csv(OUT / "s4_imputacje_glowny.csv", index=False)

# =============================================================================
# 2. PUNKT 5: PPC DLA KONTRASTU
# =============================================================================
print("\n### 2. Predykcyjna wartość p a posteriori (PPC)")


def ppc(df):
    """PPC p = odsetek imputacji, w których statystyka na danych replikowanych jest
    większa niż na danych uzupełnionych (Nguyen, Lee, Carlin 2015). Wartości bliskie 0
    lub 1 wskazują, że model imputacyjny nie odtwarza danej statystyki."""
    stat = {"beta1 - beta2 (A wobec B)": ("ocena", "ocena replikowana"),
            "odsetek Y=1, kanał A": ("Y=1 uzupełnione A", "Y=1 replikowane A"),
            "odsetek Y=1, kanał B": ("Y=1 uzupełnione B", "Y=1 replikowane B"),
            "odsetek Y=1, kanał 0": ("Y=1 uzupełnione 0", "Y=1 replikowane 0")}
    return pd.DataFrame([{"statystyka": s, "średnia uzupełnione": df[a].mean(),
                          "średnia replikowane": df[b].mean(),
                          "PPC p": (df[b] > df[a]).mean()} for s, (a, b) in stat.items()])


ppc_glowny = ppc(mi_glowny)
# Kontrola czułości miary: celowo błędny model imputacyjny bez CONTACT.
# Taki model imputuje Y bez związku z kanałem, więc PPC powinno go wykryć.
mi_zly = mi(Imputator("bez_kanalu"), M_PELNE, SEED + 1, replikuj=True)
ppc_zly = ppc(mi_zly)
t_ppc = pd.concat({"model z punktu 2 (MAR)": ppc_glowny.set_index("statystyka"),
                   "model celowo błędny (bez CONTACT)": ppc_zly.set_index("statystyka")})
print(t_ppc.round(4))
t_ppc.round(4).to_csv(OUT / "s4_t1_ppc.csv")
mi_zly[["imputacja", "ocena", "ocena replikowana"]].to_csv(OUT / "s4_ppc_model_bledny.csv", index=False)

# Wykres: punkty (wynik uzupełniony, wynik replikowany) z przekątną
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
for ax, (tyt, df, kol) in zip(axs, [("Model imputacyjny z punktu 2", mi_glowny, "#1f77b4"),
                                     ("Model celowo błędny (bez CONTACT)", mi_zly, "#d62728")]):
    a, b = df["ocena"], df["ocena replikowana"]
    lo, hi = min(a.min(), b.min()) - .05, max(a.max(), b.max()) + .05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.scatter(a, b, s=18, alpha=.7, color=kol)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="β1 − β2 na danych uzupełnionych",
           ylabel="β1 − β2 na danych replikowanych",
           title=f"{tyt}\nPPC p = {pl((b > a).mean())}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=.3)
    os_pl(ax, "x")
    os_pl(ax)
fig.text(.01, .01, f"Każdy punkt to jedna z {M_PELNE} imputacji. Chmura po obu stronach "
         "przekątnej oznacza, że model imputacyjny odtwarza kontrast A wobec B.",
         fontsize=8, color="0.3")
save("s4_w1_ppc.png", rect=(0, .05, 1, 1))

# =============================================================================
# 3. INNOWACJA 1: ANALIZA WRAŻLIWOŚCI MNAR (delta-adjustment)
# =============================================================================
print("\n### 3. Analiza wrażliwości MNAR")
# Klienci bez Y mają logit szansy zakupu przesunięty o delta względem podobnych klientów
# z Y: deltaA w kanale A, deltaB w kanale B (w kanale 0 delta = 0, czyli MAR).
# exp(delta) = ile razy szansa zakupu klienta bez Y różni się od podobnego klienta z Y.
SIATKA = np.round(np.arange(-2, 2.01, .5), 2)
rows = []
for k, (dA, dB) in enumerate([(a, b) for a in SIATKA for b in SIATKA]):
    r = rubin(*mi(imp_glowny, M_SIATKA, SEED + 1000 + k, dA, dB)[["ocena", "wariancja"]].T.values)
    rows.append({"deltaA": dA, "deltaB": dB, **wiersz("", r)})
t_siatka = pd.DataFrame(rows).drop(columns="metoda")
t_siatka["istotne"] = (t_siatka["OR dolna 95%"] > 1) | (t_siatka["OR górna 95%"] < 1)
t_siatka.round(4).to_csv(OUT / "s4_t2_mnar_siatka.csv", index=False)
print(f"Siatka {len(SIATKA)} x {len(SIATKA)}, M = {M_SIATKA}: istotnych punktów "
      f"{int(t_siatka['istotne'].sum())} z {len(t_siatka)}")

# Linie przez punkt MAR (0, 0) z gęstszym krokiem, żeby znaleźć punkt przełamania:
#   tylko kanał A (deltaB = 0), tylko kanał B (deltaA = 0), kierunki przeciwne (deltaB = -deltaA)
KROK = np.round(np.arange(-3, 3.01, .1), 2)
KIERUNKI = {"tylko kanał A (δB = 0)": (1, 0), "tylko kanał B (δA = 0)": (0, 1),
            "przeciwne (δB = −δA)": (1, -1)}
rows = []
for j, (nazwa, (ka, kb)) in enumerate(KIERUNKI.items()):
    for k, t in enumerate(KROK):
        r = rubin(*mi(imp_glowny, M_LINIE, SEED + 5000 + 100 * j + k,
                      ka * t, kb * t)[["ocena", "wariancja"]].T.values)
        rows.append({"kierunek": nazwa, "t": t, "deltaA": ka * t, "deltaB": kb * t,
                     **wiersz("", r)})
t_linie = pd.DataFrame(rows).drop(columns="metoda")
t_linie["istotne"] = (t_linie["OR dolna 95%"] > 1) | (t_linie["OR górna 95%"] < 1)
t_linie.round(4).to_csv(OUT / "s4_t3_mnar_linie.csv", index=False)

# Punkt przełamania: najmniejsze |t| po każdej stronie, od którego przedział ufności
# OR przestaje zawierać 1 (wniosek "brak istotnej różnicy A i B" się zmienia).
rows = []
for nazwa, g in t_linie.groupby("kierunek", sort=False):
    for znak, strona in [(1, "t > 0"), (-1, "t < 0")]:
        gg = g[np.sign(g["t"]) == znak].sort_values("t", key=np.abs)
        ist = gg[gg["istotne"]]
        if len(ist):
            p0 = ist.iloc[0]
            kier = "A istotnie lepszy" if p0["OR dolna 95%"] > 1 else "B istotnie lepszy"
            rows.append({"kierunek": nazwa, "strona": strona, "t": p0["t"],
                         "deltaA": p0["deltaA"], "deltaB": p0["deltaB"],
                         "exp(deltaA)": np.exp(p0["deltaA"]), "exp(deltaB)": np.exp(p0["deltaB"]),
                         "OR A wobec B": p0["OR A wobec B"], "wniosek po przełamaniu": kier})
        else:
            rows.append({"kierunek": nazwa, "strona": strona, "t": np.nan,
                         "wniosek po przełamaniu": "brak przełamania w zakresie |t| ≤ 3"})
t_przel = pd.DataFrame(rows)
print(f"Punkty przełamania (M = {M_LINIE} na punkt, krok 0,1):\n", t_przel.round(3).to_string(index=False))
t_przel.round(4).to_csv(OUT / "s4_t4_punkt_przelamania.csv", index=False)

# Mapa ciepła OR A wobec B na siatce
piv_or = t_siatka.pivot(index="deltaB", columns="deltaA", values="OR A wobec B").sort_index(ascending=False)
piv_ist = t_siatka.pivot(index="deltaB", columns="deltaA", values="istotne").sort_index(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6.4))
im = ax.imshow(np.log(piv_or.values), cmap="RdBu_r",
               norm=TwoSlopeNorm(vcenter=0, vmin=np.log(piv_or.values).min(),
                                 vmax=np.log(piv_or.values).max()))
for i in range(piv_or.shape[0]):
    for j in range(piv_or.shape[1]):
        v, ist = piv_or.iat[i, j], piv_ist.iat[i, j]
        ax.text(j, i, pl(v) + ("*" if ist else ""), ha="center", va="center", fontsize=8,
                fontweight="bold" if ist else "normal",
                color="white" if abs(np.log(v)) > .45 else "black")
        if ist:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="k", lw=1.5))
ax.add_patch(plt.Rectangle((list(piv_or.columns).index(0) - .5, list(piv_or.index).index(0) - .5),
                           1, 1, fill=False, ec="#2ca02c", lw=2.5))
ax.set_xticks(range(len(piv_or.columns)))
ax.set_xticklabels([pl(v, 1) for v in piv_or.columns])
ax.set_yticks(range(len(piv_or.index)))
ax.set_yticklabels([pl(v, 1) for v in piv_or.index])
ax.set(xlabel="δA: przesunięcie logitu szansy zakupu klientów bez Y w kanale A",
       ylabel="δB: przesunięcie w kanale B",
       title="Analiza wrażliwości MNAR: OR zakupu A wobec B")
cb = fig.colorbar(im, ax=ax, shrink=.8)
cb.set_label("OR A wobec B (skala logarytmiczna)")
cb.set_ticks(np.log([.5, .75, 1, 1.25, 1.5, 2]))
cb.set_ticklabels(["0,5", "0,75", "1", "1,25", "1,5", "2"])
fig.text(.01, .01, "Zielona ramka: MAR (δA = δB = 0). Czarna ramka i *: 95% przedział ufności OR "
         f"nie zawiera 1. Każdy punkt: {M_SIATKA} imputacji, reguły Rubina.\n"
         "exp(δ) mówi, ile razy szansa zakupu klientów bez Y różni się od podobnych klientów z Y "
         "(np. δ = 0,5 → 1,65 raza).", fontsize=8, color="0.3")
save("s4_w2_mnar_mapa.png", rect=(0, .06, 1, 1))

# Wykres linii: OR z przedziałem wzdłuż kierunków
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, (nazwa, g) in zip(axs, t_linie.groupby("kierunek", sort=False)):
    ax.fill_between(g["t"], g["OR dolna 95%"], g["OR górna 95%"], color="#1f77b4", alpha=.2,
                    label="95% przedział ufności")
    ax.plot(g["t"], g["OR A wobec B"], color="#1f77b4", lw=1.8, label="OR A wobec B")
    ax.axhline(1, color="k", lw=1)
    ax.axvline(0, color="#2ca02c", ls=":", lw=1.5, label="MAR")
    for _, p0 in t_przel[(t_przel["kierunek"] == nazwa) & t_przel["t"].notna()].iterrows():
        ax.axvline(p0["t"], color="#d62728", ls="--", lw=1)
        ax.text(p0["t"], .97, f" t = {pl(p0['t'], 1)} ", transform=ax.get_xaxis_transform(),
                color="#d62728", fontsize=8, va="top", ha="left" if p0["t"] > 0 else "right")
    ax.set_yscale("log")
    ax.set_yticks([.5, .75, 1, 1.5, 2, 3])
    ax.set_yticklabels(["0,5", "0,75", "1", "1,5", "2", "3"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set(title=nazwa, xlabel="t (przesunięcie logitu)")
    ax.grid(True, alpha=.3)
    os_pl(ax, "x")
axs[0].set_ylabel("OR zakupu A wobec B")
axs[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("OR A wobec B przy rosnącym odchyleniu od MAR; czerwona linia: punkt przełamania",
             fontweight="bold")
save("s4_w3_mnar_linie.png", rect=(0, 0, 1, .93))

# Wybrane scenariusze MNAR do wspólnej tabeli, przeliczone przy pełnym M.
# Scenariusze: umiarkowane MNAR (exp(0,5) = 1,65 raza) w przeciwnych kierunkach
# oraz najbliższy punkt przełamania w kierunku przeciwnym.
przel_przec = t_przel[(t_przel["kierunek"] == "przeciwne (δB = −δA)") & t_przel["t"].notna()]
SCEN = [(.5, -.5, "MNAR: δA = 0,5, δB = −0,5"), (-.5, .5, "MNAR: δA = −0,5, δB = 0,5")]
if len(przel_przec):
    p0 = przel_przec.iloc[przel_przec["t"].abs().argmin()]
    SCEN.append((p0["deltaA"], p0["deltaB"],
                 f"MNAR (punkt przełamania): δA = {pl(p0['deltaA'], 1)}, δB = {pl(p0['deltaB'], 1)}".replace("-", "−")))
for k, (dA, dB, nazwa) in enumerate(SCEN):
    r = rubin(*mi(imp_glowny, M_PELNE, SEED + 9000 + k, dA, dB)[["ocena", "wariancja"]].T.values)
    wyniki.append({**wiersz(nazwa, r), "deltaA": dA, "deltaB": dB})

# =============================================================================
# 4. INNOWACJA 3: IPW
# =============================================================================
print("\n### 4. IPW")
# Wagi 1 / P(Y obserwowane | X) z modelu Osoby 2 (s2_p_obserwacji.csv), stabilizowane
# odsetkiem obserwowanych Y. Wariant 2: model braku z interakcją CONTACT x AGE (istotną
# w punkcie 2, p = 0,001), bo model Osoby 2 ma tylko efekty główne.
p2 = pd.read_csv(OUT / "s2_p_obserwacji.csv")
assert len(p2) == n and (p2["r"].to_numpy() == d["r"].to_numpy()).all(), "Niezgodny plik s2_p_obserwacji"
X_r = np.column_stack([x_imp("bez_interakcji"), C0 * (AGE - AGE.mean()) / 10,
                       CA * (AGE - AGE.mean()) / 10])
b_r, _ = logit_fit(X_r, (~braki).astype(float))
p_obs_int = expit(X_r @ b_r)
P_SR = (~braki).mean()

rows_w = []
for nazwa, p_obs in [("model braku z punktu 2", p2["p_obs"].to_numpy()),
                     ("model braku z CONTACT × AGE", p_obs_int)]:
    w = P_SR / p_obs[~braki]                          # wagi stabilizowane, tylko obserwowane Y
    q1, q99 = np.percentile(w, [1, 99])
    for przyc, ww in [("bez przycinania", w), ("przycięte do 1.–99. percentyla", np.clip(w, q1, q99))]:
        b, V = logit_fit(X_DOC[~braki], y[~braki], w=ww, robust=True)
        se = np.sqrt(V[K_AB, K_AB])
        r = {"ocena": b[K_AB], "błąd": se, "dolna": b[K_AB] - 1.959964 * se,
             "górna": b[K_AB] + 1.959964 * se, "p": 2 * stats.norm.sf(abs(b[K_AB]) / se)}
        wyniki.append(wiersz(f"IPW (MAR), {nazwa}, {przyc}", r))
        rows_w.append({"model wag": nazwa, "wariant": przyc, "min": ww.min(), "Q1": np.percentile(ww, 25),
                       "mediana": np.median(ww), "Q3": np.percentile(ww, 75), "max": ww.max(),
                       "max/min": ww.max() / ww.min(), "efektywna liczebność": ww.sum() ** 2 / (ww ** 2).sum()})
t_wagi = pd.DataFrame(rows_w)
print(t_wagi.round(3).to_string(index=False))
t_wagi.round(4).to_csv(OUT / "s4_t5_ipw_wagi.csv", index=False)

# =============================================================================
# 5. INNOWACJA 4: PORÓWNANIE MODELI IMPUTACYJNYCH
# =============================================================================
print("\n### 5. Porównanie modeli imputacyjnych")
# Na początek tabeli: oficjalne wiersze MCAR i MAR Osoby 3 (s3_tabela_wynikow.csv, M = 100)
# oraz ten sam model MAR z tego przebiegu (M = 500), z którym porównujemy warianty.
s3_wiersze = s3.to_dict("records")
s3_wiersze[1]["metoda"] = "Imputacja wielokrotna (MAR), M = 100 (03_modele.py)"
wyniki[:0] = s3_wiersze + [wiersz(f"Imputacja wielokrotna (MAR), model z punktu 2, M = {M_PELNE}", r_mar)]
rows_war = [{"wariant": f"model z punktu 2 (główny), M = {M_PELNE}", **r_mar}]
for k, (war, nazwa) in enumerate([("bez_interakcji", "bez interakcji CONTACT × HOUSING_LOAN"),
                                  ("z_wiekiem", "z dodaną interakcją CONTACT × AGE")]):
    df_w = mi(Imputator(war), M_PELNE, SEED + 20 + k, replikuj=True)
    r = rubin(df_w["ocena"], df_w["wariancja"])
    r_ppc = (df_w["ocena replikowana"] > df_w["ocena"]).mean()
    rows_war.append({"wariant": nazwa, **r, "PPC p": r_ppc})
    wyniki.append(wiersz(f"Imputacja wielokrotna (MAR), {nazwa}", r))
rows_war[0]["PPC p"] = ppc_glowny.iloc[0]["PPC p"]
# Wariant MID (von Hippel 2007): imputacja, a potem usunięcie przypadków z imputowanym Y
# przed szacowaniem modelu docelowego. W naszych danych brakuje TYLKO Y, więc po usunięciu
# imputowanych Y w każdej imputacji zostaje ten sam zbiór obserwowanych Y: wszystkie M
# ocen są identyczne (B = 0), a MID jest dokładnie równy analizie kompletnych przypadków.
# Nie liczymy go więc w pętli (reguły Rubina dzieliłyby przez B = 0), tylko zapisujemy
# ten wynik wprost. To samo w sobie jest wnioskiem: ewentualna korekta w MI pochodzi
# wyłącznie z EDUCATION i interakcji w modelu imputacyjnym.
r_mid = r_cc
rows_war.append({"wariant": "MID (= kompletne przypadki, bo brakuje tylko Y)", **r_mid,
                 "FMI": np.nan, "PPC p": np.nan})
t_war = pd.DataFrame(rows_war)
t_war["OR A wobec B"] = np.exp(t_war["ocena"])
print(t_war[["wariant", "ocena", "błąd", "OR A wobec B", "p", "FMI", "PPC p"]].round(4).to_string(index=False))
t_war.round(4).to_csv(OUT / "s4_t6_warianty_imputacji.csv", index=False)
wyniki.append(wiersz("MID (MAR), równy kompletnym przypadkom", r_mid, fmi=False))

# =============================================================================
# 6. INNOWACJA 5: EFEKT NA SKALI PRAWDOPODOBIEŃSTWA
# =============================================================================
print("\n### 6. Średni efekt krańcowy A wobec B (punkty procentowe)")
ame_cc, var_ame_cc = ame_ab(b_cc, V_cc)
r_ame = rubin(mi_glowny["AME"], mi_glowny["AME wariancja"])
t_ame = pd.DataFrame([
    {"metoda": "Kompletne przypadki (MCAR)", "AME (pp)": 100 * ame_cc,
     "błąd (pp)": 100 * np.sqrt(var_ame_cc),
     "dolna 95% (pp)": 100 * (ame_cc - 1.959964 * np.sqrt(var_ame_cc)),
     "górna 95% (pp)": 100 * (ame_cc + 1.959964 * np.sqrt(var_ame_cc)),
     "p": 2 * stats.norm.sf(abs(ame_cc) / np.sqrt(var_ame_cc))},
    {"metoda": "Imputacja wielokrotna (MAR)", "AME (pp)": 100 * r_ame["ocena"],
     "błąd (pp)": 100 * r_ame["błąd"], "dolna 95% (pp)": 100 * r_ame["dolna"],
     "górna 95% (pp)": 100 * r_ame["górna"], "p": r_ame["p"]}])
print(t_ame.round(3).to_string(index=False))
t_ame.round(4).to_csv(OUT / "s4_t7_efekt_prawdopodobienstwo.csv", index=False)

# =============================================================================
# 7. INNOWACJA 6: TEST NA SZTUCZNYCH BRAKACH
# =============================================================================
print(f"\n### 7. Test na sztucznych brakach ({R_SYM} powtórzeń, M = {M_SYM})")
# Symulacja ze znaną prawdą. Cechy X wszystkich 11 304 klientów zostają bez zmian,
# a Y losujemy z "prawdziwego" modelu dopasowanego do obserwowanych Y. Prawdziwy model
# jest celowo bogatszy niż model imputacyjny (wiek w przedziałach i interakcja CONTACT x
# wiek w przedziałach), żeby sprawdzić imputację w realistycznej sytuacji, gdy model
# imputacyjny nie jest idealny. Następnie usuwamy część Y znanym mechanizmem MAR,
# liczymy kompletne przypadki (CC) i imputację wielokrotną modelem z punktu 2,
# i porównujemy z wartością prawdziwą: obciążenie, RMSE, pokrycie 95% przedziałów.
#   Mechanizm 1 (jak w danych): brak zależy od CONTACT x AGE (model braku z części 4)
#   Mechanizm 2 (silny MAR przez cechę pomocniczą): brak zależy od EDUCATION, najsilniej
#     w kanale B; EDUCATION wiąże się z Y i nie ma jej w modelu docelowym
# Przedziały wieku 18-30, 31-40, 41-50, 51+ (w kanale 0 są tylko 4 osoby 61+, wszystkie
# bez zakupu, więc osobny przedział 61+ dawałby separację w modelu)
wiek_p = pd.cut(d["age"], [17, 30, 40, 50, 100]).cat.codes.to_numpy()
WB = np.column_stack([(wiek_p == k).astype(float) for k in range(1, 4)])   # ref: 18-30
X_PRAWDA = np.column_stack([x_imp("glowny"), WB, C0[:, None] * WB, CA[:, None] * WB])
b_prawda, _ = logit_fit(X_PRAWDA[~braki], y[~braki])
p_prawda = expit(X_PRAWDA @ b_prawda)
# Wartość prawdziwa = beta1 - beta2 modelu docelowego w "populacji": MLE z oczekiwanym
# Y (p_prawda) zamiast losowego Y, czyli rzut prawdziwego modelu na model docelowy.
prawda = logit_fit(X_DOC, p_prawda)[0][K_AB]
X_IMP_G = x_imp("glowny")
p_mech1 = 1 - p_obs_int                                   # ok. 25% braków
EDU_NISKIE = EDU[:, 0] + EDU[:, 2]                       # podstawowe lub nieznane
eta2 = -1.6 + 1.2 * EDU_NISKIE - .8 * EDU[:, 1] + 1.0 * isB * EDU_NISKIE
p_mech2 = expit(eta2)
rows = []
rng = np.random.default_rng(SEED + 40)
for mech, p_m in [("mechanizm 1: jak w danych (CONTACT × AGE)", p_mech1),
                  ("mechanizm 2: silny MAR przez EDUCATION", p_mech2)]:
    for rep in range(R_SYM):
        ys = rng.binomial(1, p_prawda).astype(float)       # nowe "pełne" dane
        mis = rng.random(n) < p_m                          # sztuczne braki
        b, V = logit_fit(X_DOC[~mis], ys[~mis])            # kompletne przypadki
        se = np.sqrt(V[K_AB, K_AB])
        rows.append({"mechanizm": mech, "metoda": "kompletne przypadki", "powtórzenie": rep,
                     "ocena": b[K_AB], "dolna": b[K_AB] - 1.959964 * se,
                     "górna": b[K_AB] + 1.959964 * se, "odsetek braków": mis.mean()})
        bi, Vi = logit_fit(X_IMP_G[~mis], ys[~mis])        # imputacja modelem z punktu 2
        oc, wa = [], []
        for _ in range(M_SYM):
            beta = rng.multivariate_normal(bi, Vi)
            yy = ys.copy()
            yy[mis] = rng.binomial(1, expit(X_IMP_G[mis] @ beta))
            bb, VV = logit_fit(X_DOC, yy)
            oc.append(bb[K_AB])
            wa.append(VV[K_AB, K_AB])
        r = rubin(oc, wa)
        rows.append({"mechanizm": mech, "metoda": "imputacja wielokrotna", "powtórzenie": rep,
                     "ocena": r["ocena"], "dolna": r["dolna"], "górna": r["górna"],
                     "odsetek braków": mis.mean()})
sym = pd.DataFrame(rows)
sym["pokrycie"] = (sym["dolna"] <= prawda) & (prawda <= sym["górna"])
sym["szerokość"] = sym["górna"] - sym["dolna"]
sym["błąd kw."] = (sym["ocena"] - prawda) ** 2
t_sym = sym.groupby(["mechanizm", "metoda"], sort=False).agg(
    odsetek_brakow=("odsetek braków", "mean"), srednia_ocena=("ocena", "mean"),
    pokrycie_95=("pokrycie", "mean"), szerokosc_przedzialu=("szerokość", "mean"),
    mse=("błąd kw.", "mean"), sd=("ocena", "std")).reset_index()
t_sym["obciążenie"] = t_sym["srednia_ocena"] - prawda
t_sym["błąd MC obciążenia"] = t_sym.pop("sd") / np.sqrt(R_SYM)
t_sym["RMSE"] = np.sqrt(t_sym.pop("mse"))
t_sym.insert(2, "wartość prawdziwa", prawda)
print(t_sym.round(4).to_string(index=False))
t_sym.round(4).to_csv(OUT / "s4_t8_sztuczne_braki.csv", index=False)
sym.to_csv(OUT / "s4_sztuczne_braki_powtorzenia.csv", index=False)

# =============================================================================
# 8. WSPÓLNA TABELA WYNIKÓW
# =============================================================================
print("\n### 8. Tabela wyników (format jak s3_tabela_wynikow.csv)")
t_wyn = pd.DataFrame(wyniki)
kol = ["metoda", "beta1 - beta2", "błąd", "OR A wobec B", "OR dolna 95%", "OR górna 95%", "p", "FMI"]
t_wyn = t_wyn[kol]
print(t_wyn.round(4).to_string(index=False))
t_wyn.to_csv(OUT / "s4_tabela_wynikow.csv", index=False)

# Wykres leśny wszystkich metod (do wykorzystania przez Osobę 5 w punkcie 4)
fig, ax = plt.subplots(figsize=(9, .42 * len(t_wyn) + 1.2))
yy = np.arange(len(t_wyn))[::-1]
kolory = ["#555555" if "Kompletne" in m else "#d62728" if "MNAR" in m else
          "#2ca02c" if "IPW" in m else "#1f77b4" for m in t_wyn["metoda"]]
for yi, (_, w), k in zip(yy, t_wyn.iterrows(), kolory):
    ax.plot([w["OR dolna 95%"], w["OR górna 95%"]], [yi, yi], color=k, lw=1.5)
ax.scatter(t_wyn["OR A wobec B"], yy, color=kolory, zorder=3, s=30)
ax.axvline(1, color="k", lw=1)
ax.set_yticks(yy)
ax.set_yticklabels(t_wyn["metoda"], fontsize=8)
ax.set_xscale("log")
ax.set_xticks([.5, .75, 1, 1.5, 2, 3])
ax.set_xticklabels(["0,5", "0,75", "1", "1,5", "2", "3"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set(xlabel="OR zakupu A wobec B (95% przedział ufności)",
       title="Iloraz szans zakupu A wobec B według metody")
ax.xaxis.grid(True, alpha=.3)
save("s4_w4_wykres_lesny.png")
print(f"\nCzas obliczeń: {time.time() - T0:.0f} s. Pliki s4_* zapisane w: {OUT}")
