# =============================================================================
# Punkt 3: model regresji logistycznej przy założeniach MCAR oraz MAR
# Lokalizacja danych i folderu wyników jest w 00_setup.py
# Pliki wynikowe tej części mają przedrostek s3_.
# =============================================================================
import sys
import importlib
import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix

S = importlib.import_module("00_setup")
pd.set_option("display.width", 160)
print("Wersje: Python", sys.version.split()[0], "| pandas", pd.__version__,
      "| numpy", np.__version__, "| scipy", scipy.__version__,
      "| statsmodels", statsmodels.__version__)

d = S.load_clean()
obs = d[d["r"] == 0]  # tylko klienci z obserwowanym Y
OUT = S.WYNIKI

# Model docelowy z zadania. Kategorie referencyjne (z 00_setup):
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
rng = np.random.default_rng(S.SEED)


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
