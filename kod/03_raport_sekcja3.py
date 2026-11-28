# =============================================================================
# Sekcja 3 raportu (modele przy MCAR i MAR) -> raport/03_raport_czesc3.docx
# Uruchamiać PO 03_modele.py: liczby są czytane z plików s3_*.csv w wyniki/.
# Wygląd jak w części 1 (moduł raport_styl.py). Wymaga: python-docx.
# =============================================================================
import importlib
import pandas as pd
from raport_styl import Raport, f, fp, S

W = S.WYNIKI
# Numeracja ciągła w całym raporcie: części 1-2 kończą się na tabeli 10 i rysunku 8
R = Raport(nr_tab=11, nr_rys=9)
akapit, tabela = R.akapit, R.tabela


# --- dane z wyników 03_modele.py
cc = pd.read_csv(W / "s3_model_mcar.csv", index_col=0)
kontr = pd.read_csv(W / "s3_kontrola_kontrastu.csv", index_col=0)["wartość"]
rub = pd.read_csv(W / "s3_reguly_rubina.csv", index_col=0)["wartość"]
spr = pd.read_csv(W / "s3_odsetek_y_imputowane.csv", index_col=0)
wyn = pd.read_csv(W / "s3_tabela_wynikow.csv")
mcar, mar = wyn.iloc[0], wyn.iloc[1]
M = int(rub["m (liczba imputacji)"])
# odsetek klientów kanału B z sukcesem w poprzedniej kampanii: z Y i bez Y
dane = S.load_clean()
kanal_b = dane[dane["contact"] == "2"]
sukces_b = kanal_b.groupby("r")["poutcome"].apply(lambda s: 100 * (s == "success").mean())
n_obs, n_brak = int((dane["r"] == 0).sum()), int((dane["r"] == 1).sum())

# =============================================================================
R.tytul("Część 3: model regresji logistycznej przy założeniach MCAR i MAR")

# --- 3.1 Model docelowy
R.tytul("1. Model docelowy")
akapit("Szacujemy model logitowy z treści zadania. Po lewej stronie wzoru jest logit "
       "prawdopodobieństwa zakupu, czyli logarytm szansy:")
akapit("logit P(Y_NEWDEPOSIT = 1 | X) = β0 + β1·CONTACT_A + β2·CONTACT_B + β3·AGE + "
       "β4·HOUSING_LOAN + β5·POUTCOME_failure + β6·POUTCOME_success + β7·POUTCOME_other")
akapit("Pytanie zadania dotyczy różnicy β1 − β2, czyli porównania kanału A z kanałem B. "
       "Iloraz szans OR = exp(β1 − β2) mówi, ile razy szansa zakupu w kanale A jest większa "
       "niż w kanale B przy tych samych pozostałych cechach klienta.")
akapit("Kodowanie cech jest takie jak w przykładowym kodzie z zadania: dla CONTACT poziomem "
       "odniesienia jest kanał B (2), dla HOUSING_LOAN – posiadanie kredytu (yes), dla "
       "POUTCOME – unknown. Przy kanale B jako odniesieniu współczynnik przy kanale A jest od "
       "razu równy β1 − β2. Grupa bez komunikacji (0) zostaje w modelu, bo pomaga oszacować "
       "wpływ pozostałych cech.")

# --- 3.2 MCAR
R.tytul("2. Model przy założeniu MCAR – analiza kompletnych przypadków")
akapit(f"Przy założeniu MCAR braki są czysto przypadkowe, więc klienci z obserwowanym Y są "
       f"losową częścią całej próby. Wystarczy wtedy usunąć {n_brak} klientów bez Y i "
       f"oszacować model na pozostałych {n_obs}. Oszacowania nie są obciążone, tracimy tylko "
       f"część informacji (szersze przedziały ufności).")

nazwy = {"Intercept": "Wyraz wolny",
         "contact[T.0]": "Brak komunikacji (wobec kanału B)",
         "contact[T.1]": "Kanał A (wobec kanału B) = β1 − β2",
         "house_loan[T.no]": "Brak kredytu mieszkaniowego (wobec: ma kredyt)",
         "poutcome[T.failure]": "POUTCOME: failure (wobec unknown)",
         "poutcome[T.success]": "POUTCOME: success (wobec unknown)",
         "poutcome[T.other]": "POUTCOME: other (wobec unknown)",
         "age": "Wiek (1 rok)"}
t = pd.DataFrame({"Parametr": [nazwy[i] for i in cc.index],
                  "Ocena": [f(x, 3) for x in cc["beta"]],
                  "Błąd": [f(x, 3) for x in cc["SE"]],
                  "p": [fp(x) for x in cc["p"]],
                  "OR": ["–" if i == "Intercept" else f(x) for i, x in zip(cc.index, cc["OR"])]})
tabela(t, f"Model docelowy na kompletnych przypadkach (n = {n_obs}).")

akapit(f"Kanał A ma szansę zakupu {f(mcar['OR A wobec B'])} raza większą niż kanał B "
       f"(95% PU: {f(mcar['OR dolna 95%'])}–{f(mcar['OR górna 95%'])}, p = {fp(mcar['p'])}). "
       f"Przedział zawiera 1, więc różnica między kanałami nie jest istotna statystycznie. "
       f"Pozostałe cechy działają zgodnie z oczekiwaniami: klienci bez komunikacji mają szansę "
       f"zakupu około dwa razy mniejszą niż w kanale B, sukces w poprzedniej kampanii zwiększa "
       f"szansę kilkunastokrotnie, a brak kredytu mieszkaniowego – prawie dwukrotnie.")
akapit(f"Kontrola kontrastu: ten sam model z poziomem odniesienia „brak komunikacji” (jak we "
       f"wzorze z zadania) daje β1 = {f(kontr.iloc[0], 3)} i β2 = {f(kontr.iloc[1], 3)}. "
       f"Różnica β1 − β2 = {f(kontr.iloc[3], 3)}, a jej błąd policzony ze wzoru "
       f"Var(β1) + Var(β2) − 2·Cov(β1, β2) wynosi {f(kontr.iloc[4], 3)}. Oba kodowania dają "
       f"identyczny wynik, więc kontrast A wobec B jest policzony poprawnie.")

# --- 3.3 MAR
R.tytul("3. Model przy założeniu MAR – imputacja wielokrotna")
akapit("Założenie MAR mówi, że brak Y może zależeć od obserwowanych cech klienta, ale nie od "
       "samego zakupu. Jest słabsze niż MCAR (MCAR jest jego szczególnym przypadkiem), więc "
       "imputacja przy MAR jest poprawna także wtedy, gdy braki są całkowicie losowe. "
       "Imputacja może dać inny wynik niż analiza kompletnych przypadków tylko dzięki "
       "informacji, której nie ma w modelu docelowym. W części 2 takimi cechami okazały się "
       "EDUCATION i interakcja CONTACT × HOUSING_LOAN – obie wiążą się z zakupem.")
akapit("Brakujące Y uzupełniamy imputacją wielokrotną. Specyfikacja modelu imputacyjnego "
       "pochodzi z tabeli decyzyjnej w części 2:")

spec = pd.DataFrame({
    "Element": ["Zmienna imputowana", "Wzorzec braków", "Metoda", "Predyktory", "Interakcje",
                "Poza modelem", "Liczba imputacji", "Ziarno losowe", "Łączenie wyników"],
    "Specyfikacja": [
        f"Y_NEWDEPOSIT – jedyna zmienna z brakami ({n_brak} obs., 25,5%)",
        "monotoniczny (brakuje tylko Y), więc wystarcza jeden krok imputacji bez iteracji",
        "regresja logistyczna z losowaniem parametrów",
        "CONTACT, AGE, HOUSING_LOAN, POUTCOME, EDUCATION (cecha pomocnicza)",
        "CONTACT × HOUSING_LOAN",
        "wskaźnik braku M (w kodzie r)",
        f"m = {M}; efektywność (1 + FMI/m)⁻¹ = {f(rub['efektywność'], 3)}",
        str(S.SEED),
        "reguły Rubina dla β1 − β2 i jego błędu; exp liczone na końcu"]})
tabela(spec, "Specyfikacja modelu imputacyjnego.", num_od=2)

akapit(f"Jedna imputacja przebiega w trzech krokach. (1) Model imputacyjny jest dopasowany na "
       f"klientach z Y, a z rozkładu jego oszacowań losujemy nowy zestaw współczynników – "
       f"współczynniki nie są znane dokładnie, więc każda imputacja używa trochę innej ich "
       f"wersji. (2) Z wylosowanymi współczynnikami liczymy dla każdego klienta bez Y "
       f"prawdopodobieństwo zakupu i losujemy Y = 1 z tym prawdopodobieństwem. (3) Na "
       f"uzupełnionym zbiorze szacujemy model docelowy i zapisujemy β1 − β2 oraz jego błąd. "
       f"Kroki powtarzamy {M} razy, a wyniki łączymy regułami Rubina.")

t = pd.DataFrame({"Kanał": ["B", "brak komunikacji", "A"],
                  "Obserwowane Y": [f"{100 * spr.loc[k, 'obserwowane Y']:.1f}%".replace(".", ",")
                                    for k in [2, 0, 1]],
                  "Imputowane Y (średnio)": [f"{100 * spr.loc[k, 'imputowane Y']:.1f}%".replace(".", ",")
                                             for k in [2, 0, 1]]})
tabela(t, "Odsetek klientów z Y = 1 wśród obserwowanych i imputowanych.")
akapit(f"W kanale A i w grupie bez komunikacji odsetek zakupów wśród imputowanych jest taki "
       f"sam jak wśród obserwowanych. W kanale B jest niższy i nie jest to błąd imputacji. "
       f"Klienci kanału B bez Y rzadziej mieli sukces w poprzedniej kampanii ({f(sukces_b[1], 1)}% "
       f"wobec {f(sukces_b[0], 1)}% wśród klientów B z Y) i częściej mają kredyt mieszkaniowy – obie cechy obniżają szansę "
       "zakupu, więc model przewiduje im mniej zakupów. Te same cechy są w modelu docelowym, "
       "dlatego ta różnica prawie nie zmienia porównania kanału A z kanałem B.")

t = pd.DataFrame({"Miara": ["β1 − β2 (średnia z imputacji)", "Wariancja wewnątrz imputacji",
                            "Wariancja między imputacjami", "Wariancja całkowita",
                            "Błąd standardowy", "Stopnie swobody",
                            "FMI (frakcja utraconej informacji)"],
                  "Wartość": [f(rub["ocena beta1 - beta2"], 4),
                              f(rub["wariancja wewnątrz imputacji"], 5),
                              f(rub["wariancja między imputacjami"], 5),
                              f(rub["wariancja całkowita"], 5),
                              f(rub["błąd standardowy"], 4),
                              f(rub["stopnie swobody"], 0),
                              f(rub["FMI"], 3)]})
tabela(t, f"Połączenie wyników z {M} imputacji regułami Rubina.")
akapit(f"Wariancja całkowita składa się z dwóch części: zwykłej niepewności wynikającej z "
       f"próby (wariancja wewnątrz imputacji) oraz dodatkowej niepewności z powodu braków "
       f"(wariancja między imputacjami). FMI = {f(mar['FMI'])} oznacza, że braki odebrały "
       f"około {round(100 * mar['FMI'])}% informacji o różnicy między kanałami, czyli mniej "
       f"więcej tyle, ile wynosi odsetek braków Y. Kanał A ma szansę zakupu {f(mar['OR A wobec B'])} raza większą "
       f"niż kanał B (95% PU: {f(mar['OR dolna 95%'])}–{f(mar['OR górna 95%'])}, "
       f"p = {fp(mar['p'])}).")

# --- 3.4 Porównanie
R.tytul("4. Porównanie wyników przy MCAR i MAR")
t = pd.DataFrame({"Metoda": wyn["metoda"],
                  "β1 − β2": [f(x, 3) for x in wyn["beta1 - beta2"]],
                  "Błąd": [f(x, 3) for x in wyn["błąd"]],
                  "OR A wobec B": [f(x) for x in wyn["OR A wobec B"]],
                  "95% PU dla OR": [f"{f(a)}–{f(b)}" for a, b in
                                    zip(wyn["OR dolna 95%"], wyn["OR górna 95%"])],
                  "p": [fp(x) for x in wyn["p"]],
                  "FMI": ["–" if pd.isna(x) else f(x) for x in wyn["FMI"]]})
tabela(t, "Iloraz szans zakupu: kanał A wobec kanału B.")
akapit("Obie metody dają ten sam wniosek: szansa zakupu w kanale A jest około 1,2 raza "
       "większa niż w kanale B, ale różnica nie jest istotna statystycznie. Imputacja tylko "
       "nieznacznie zmienia ocenę i błąd.")
akapit("Wyniki są do siebie zbliżone, bo braki Y prawie nie zależą od cech klientów, a cechy "
       "silnie związane z zakupem są już w modelu docelowym. Za wynik główny przyjmujemy model "
       "przy MAR, bo opiera się na słabszym założeniu i wykorzystuje dodatkowe informacje "
       "(EDUCATION, CONTACT × HOUSING_LOAN). Analiza kompletnych przypadków służy jako punkt "
       "odniesienia. Oba wyniki zakładają, że brak Y nie zależy od samego zakupu – tego dane "
       "nie pozwalają sprawdzić.")

R.zapisz("03_raport_czesc3.docx")
