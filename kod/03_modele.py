# =============================================================================
# Punkt 3: model regresji logistycznej przy założeniach MCAR oraz MAR
# Lokalizacja danych i folderu wyników jest w 00_setup.py
# =============================================================================
import importlib
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix

S = importlib.import_module("00_setup")
pd.set_option("display.width", 160)

d = S.load_clean()
obs = d[d["r"] == 0]  # tylko klienci z obserwowanym Y

# Model docelowy z zadania. Kategorie referencyjne (z 00_setup):
# contact = 2 (kanał B), house_loan = yes, poutcome = unknown
F_DOCELOWY = "y ~ contact + age + house_loan + poutcome"

# -----------------------------------------------------------------------------
# 1. MCAR - analiza kompletnych przypadków
# -----------------------------------------------------------------------------
print("### 1. Model docelowy przy MCAR (kompletne przypadki)")
m_cc = smf.logit(F_DOCELOWY, data=obs).fit(disp=0)
print(m_cc.summary())

# Przy ref = 2 współczynnik contact[T.1] to od razu beta1 - beta2 (A wobec B)
est_cc = m_cc.params["contact[T.1]"]
se_cc = m_cc.bse["contact[T.1]"]
ci_cc = m_cc.conf_int().loc["contact[T.1]"]
print(f"\nbeta1 - beta2 = {est_cc:.4f}, SE = {se_cc:.4f}, p = {m_cc.pvalues['contact[T.1]']:.4f}")
print(f"OR A wobec B = {np.exp(est_cc):.4f}, 95% CI: {np.exp(ci_cc[0]):.4f} - {np.exp(ci_cc[1]):.4f}")

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

# -----------------------------------------------------------------------------
# 2. MAR - model imputacyjny (specyfikacja z punktu 2)
# -----------------------------------------------------------------------------
print("\n### 2. Model imputacyjny dla Y")
F_IMP = "contact + age + house_loan + poutcome + education + contact:house_loan"
X = dmatrix(F_IMP, d, return_type="dataframe")  # macierz cech dla wszystkich klientów
braki = d["r"] == 1

m_imp = sm.Logit(d.loc[~braki, "y"], X[~braki]).fit(disp=0)
print(m_imp.summary())

rng = np.random.default_rng(S.SEED)


def imputuj():
    """Jedna imputacja: losujemy parametry modelu, a potem brakujące Y."""
    beta = rng.multivariate_normal(m_imp.params, m_imp.cov_params())
    p = 1 / (1 + np.exp(-(X[braki] @ beta)))
    d_imp = d.copy()
    d_imp.loc[braki, "y"] = rng.binomial(1, p)
    return d_imp


# -----------------------------------------------------------------------------
# 3. MAR - 50 imputacji, model docelowy w każdej, łączenie regułami Rubina
# -----------------------------------------------------------------------------
print("\n### 3. Imputacja wielokrotna")
M = 100
oceny, bledy, odsetki, imputacje = [], [], [], []
for i in range(M):
    d_imp = imputuj()
    imputacje.append(d_imp.loc[braki, "y"].to_numpy())
    m = smf.logit(F_DOCELOWY, data=d_imp).fit(disp=0)
    oceny.append(m.params["contact[T.1]"])
    bledy.append(m.bse["contact[T.1]"])
    odsetki.append(d_imp[braki].groupby("contact", observed=True)["y"].mean())

# Sprawdzenie: odsetek Y = 1 według kanału, obserwowane vs imputowane (średnio z 50)
spr = pd.DataFrame({
    "obserwowane Y": obs.groupby("contact", observed=True)["y"].mean(),
    "imputowane Y": pd.concat(odsetki, axis=1).mean(axis=1),
})
print("Odsetek Y = 1 według kanału:")
print(spr.round(4))

# Reguły Rubina
oceny = np.array(oceny)
bledy = np.array(bledy)
Q = oceny.mean()                              # (38) ocena punktowa
W = (bledy ** 2).mean()                       # (39) wariancja wewnątrzimputacyjna
B = oceny.var(ddof=1)                         # (40) wariancja międzyimputacyjna
T = W + (1 + 1 / M) * B                       # (41) wariancja całkowita
se_mi = np.sqrt(T)                            # (42) błąd standardowy
gamma = (1 + 1 / M) * B / W                   # (46) względny wzrost wariancji
v = (M - 1) * (1 + 1 / gamma) ** 2            # (44) stopnie swobody
fmi = (gamma + 2 / (v + 3)) / (gamma + 1)     # (47) frakcja brakującej informacji
efekt = 1 / (1 + fmi / M)                     # (48) efektywność

t_kr = stats.t.ppf(0.975, v)
ci_mi = (Q - t_kr * se_mi, Q + t_kr * se_mi)
p_mi = 2 * stats.t.sf(abs(Q / se_mi), v)

print(f"\nbeta1 - beta2 = {Q:.4f}, SE = {se_mi:.4f}, p = {p_mi:.4f}")
print(f"wariancja wewnątrz = {W:.5f}, między = {B:.5f}, całkowita = {T:.5f}")
print(f"wzrost wariancji = {gamma:.4f}, df = {v:.1f}, FMI = {fmi:.4f}, efektywność = {efekt:.4f}")
print(f"OR A wobec B = {np.exp(Q):.4f}, 95% CI: {np.exp(ci_mi[0]):.4f} - {np.exp(ci_mi[1]):.4f}")

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
# 5. Zapis wyników (tabele do raportu i imputacje dla diagnostyki w punkcie 5)
# -----------------------------------------------------------------------------
OUT = S.WYNIKI
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

# imputowane wartości Y: wiersz = klient z brakiem Y (indeks z dane_czyste), kolumna = imputacja
pd.DataFrame(np.column_stack(imputacje), index=d.index[braki],
             columns=[f"imp_{i}" for i in range(1, M + 1)]).to_csv(OUT / "s3_imputacje.csv")
print("\nZapisano pliki s3_*.csv w", OUT)
