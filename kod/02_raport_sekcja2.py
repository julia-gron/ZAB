# =============================================================================
# Część 2 raportu (mechanizm braków) -> raport/02_raport_czesc2.docx
# Uruchamiać PO 02_mechanizm.py: liczby są czytane z plików s2_*.csv w wyniki/,
# więc raport zgadza się z wydrukiem. Wygląd (marginesy, czcionki, odstępy) jest
# brany z raportu części 1 (raport/01_raport_czesc1.docx), żeby obie części były spójne.
# Numeracja tabel i rysunków jest kontynuacją części 1 (tabele 1-3, rysunki 1-5).
# Wymaga pakietu python-docx (pip install python-docx).
# =============================================================================
import importlib
import pandas as pd
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

S = importlib.import_module("00_setup")
W = S.WYNIKI
RAPORT = S.BASE / "raport"
SZABLON = RAPORT / "01_raport_czesc1.docx"
NR_TAB, NR_RYS = 4, 6                                     # pierwsze numery w tej części


def f(x, n=2):
    """Liczba w zapisie polskim."""
    return f"{x:.{n}f}".replace(".", ",")


def fp(p):
    """Wartość p: '< 0,001' albo trzy miejsca po przecinku."""
    return "< 0,001" if p < .001 else f(p, 3)


def pr(x):
    """Odsetek (0-1) jako procent z jednym miejscem po przecinku."""
    return f(100 * x, 1) + "%"


# Dokument na bazie raportu części 1: zachowujemy style i ustawienia strony, czyścimy treść
doc = Document(SZABLON)
body = doc.element.body
for el in list(body):
    if not el.tag.endswith("sectPr"):
        body.remove(el)


def akapit(tekst):
    p = doc.add_paragraph(tekst)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(12)
    return p


def tytul(tekst):
    """Nagłówek w stylu części 1: zwykły tekst w osobnym akapicie."""
    doc.add_paragraph(tekst)


def tabela(df, podpis, num_od=1, num_do=None):
    """Podpis nad tabelą, tabela z siatką; liczby wyrównane do prawej (jak w części 1)."""
    global NR_TAB
    doc.add_paragraph(f"Tabela {NR_TAB}. {podpis}")
    NR_TAB += 1
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    for i, c in enumerate(df.columns):
        t.rows[0].cells[i].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    for r_i, row in enumerate(t.rows):
        for c_i, c in enumerate(row.cells):
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                if r_i == 0:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                elif num_od <= c_i < (num_do or len(df.columns)):   # kolumny liczbowe
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                for r in p.runs:
                    r.font.name = "Aptos Narrow"
                    r.font.size = Pt(11)
                    r.bold = r_i == 0
    doc.add_paragraph()


def rysunek(plik, podpis, szer=16):
    """Podpis nad rysunkiem, jak w części 1."""
    global NR_RYS
    doc.add_paragraph(f"Rysunek {NR_RYS}. {podpis}")
    NR_RYS += 1
    doc.add_picture(str(W / plik), width=Cm(szer))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


# --- wyniki z 02_mechanizm.py
ttest = pd.read_csv(W / "s2_t1_test_t_wiek.csv")
kat = pd.read_csv(W / "s2_t2_braki_wg_kategorii.csv")
chi = pd.read_csv(W / "s2_t3_chi2_braki.csv", index_col=0)
t3r = pd.read_csv(W / "s2_t4_model_braku_type3.csv", index_col=0)
orr = pd.read_csv(W / "s2_t5_model_braku_or.csv", index_col=0)
wk = pd.read_csv(W / "s2_t6_braki_wiek_kanal.csv")
t3y = pd.read_csv(W / "s2_t7_model_y_type3.csv", index_col=0)
ory = pd.read_csv(W / "s2_t8_model_y_or.csv", index_col=0)
zk = pd.read_csv(W / "s2_t9_zakup_wiek_kanal.csv")
glob = pd.read_csv(W / "s2_t10_testy_globalne.csv", index_col=0)
dec = pd.read_csv(W / "s2_t11_tabela_decyzyjna.csv")
spec = pd.read_csv(W / "s2_t12_projekt_modelu_imputacyjnego.csv")

gr, gy = glob.iloc[0], glob.iloc[1]
dr = dec.set_index("element")
B = wk[wk.kanal == "B: kanał B"].set_index("wiek_p")["odsetek"]
BZ = zk[zk.kanal == "B: kanał B"].set_index("wiek_p")["odsetek"]

# =============================================================================
tytul("Część 2: mechanizm braków danych")
akapit("W tej części sprawdzamy, czy prawdopodobieństwo braku wartości Y zależy od pozostałych "
       "cech i czy istnieją cechy, za pomocą których imputacja może skorygować oszacowania "
       "(punkt 2 zadania). Braki występują tylko w Y (2 879 z 11 304 klientów, 25,47%). "
       "Wskaźnik braku oznaczamy, jak na wykładzie, przez M (M = 1: brak Y, M = 0: Y "
       "obserwowane); w kodzie jest to zmienna r. Cecha nadaje się do korekty oszacowań, gdy "
       "jest związana jednocześnie z brakiem Y i z samym Y.")

# 1.
tytul("1. Ocena losowości braków danych")
a0, a1 = ttest.iloc[0], ttest.iloc[1]
akapit("Losowość braków oceniamy dwiema metodami z wykładu. W metodzie I dzielimy zbiór według "
       "wskaźnika M na część z obserwowanym Y i część z brakiem Y i testem t porównujemy średni "
       "wiek w obu grupach. Średni wiek wynosi "
       f"{f(a0['średnia wieku'])} roku przy obserwowanym Y i {f(a1['średnia wieku'])} roku przy "
       f"braku Y (t = {f(a0['t'], 3)}, p = {fp(a0['p'])}), więc różnica nie jest istotna "
       "(tabela 4).")
t = ttest[["grupa", "n", "średnia wieku", "odch. std."]].copy()
t["n"] = t["n"].map(lambda x: f"{int(x):,}".replace(",", " "))
t["średnia wieku"], t["odch. std."] = t["średnia wieku"].map(f), t["odch. std."].map(f)
tabela(t, f"Średni wiek według wskaźnika braku Y (test t: t = {f(a0['t'], 3)}, p = {fp(a0['p'])}).")

akapit("W metodzie II sprawdzamy związek wskaźnika M z cechami skokowymi testem chi-kwadrat "
       "niezależności i miarą asocjacji V Cramera. Wiek dzielimy w tym celu na przedziały. "
       "Odsetek braków Y jest podobny we wszystkich kategoriach wszystkich cech i bliski "
       "średniej 25,5% (rysunek 6). Żaden test nie wykazuje istotnego związku, a V Cramera "
       "nie przekracza 0,03 (tabela 5). Najwyższy odsetek braków ma kategoria education = "
       f"unknown ({pr(kat[(kat.cecha == 'EDUCATION') & (kat.kategoria == 'unknown')]['odsetek braków'].iloc[0])}), "
       "ale jest to mała grupa (492 klientów) i test dla całej cechy nie jest istotny.")
t = chi.reset_index()
t["chi2"], t["df"], t["p"] = t["chi2"].map(f), t["df"].astype(int), t["p"].map(fp)
t["V Cramera"] = t["V Cramera"].map(lambda x: f(x, 3))
t.columns = ["Cecha", "Chi-kwadrat", "df", "p", "V Cramera"]
tabela(t, "Test chi-kwadrat niezależności wskaźnika braku Y od cech skokowych.")
rysunek("s2_w1_braki_wg_cech.png", "Odsetek braków Y w kategoriach cech.")

# 2.
tytul("2. Model logistyczny prawdopodobieństwa obserwacji Y")
akapit("Wszystkie cechy jednocześnie uwzględnia model regresji logistycznej, w którym, jak na "
       "slajdzie o ocenie skłonności do odpowiedzi, modelujemy prawdopodobieństwo obserwacji Y, "
       "czyli P(M = 0), względem cech CONTACT, AGE, HOUSING_LOAN, POUTCOME i EDUCATION. Test "
       f"ilorazu wiarygodności dla całego modelu daje chi-kwadrat = {f(gr['LR chi2'])} przy "
       f"{int(gr['df'])} stopniach swobody (p = {fp(gr['p'])}), a żadna cecha nie jest istotna "
       "w testach Walda (tabela 6). Statystyka c (pole pod krzywą ROC) wynosi "
       f"{f(gr['c (AUC)'], 3)}, czyli model prawie nie odróżnia klientów z brakiem Y od "
       "pozostałych (0,5 oznacza brak zdolności rozróżniania).")

t = pd.DataFrame({
    "Cecha": ["CONTACT", "AGE", "HOUSING_LOAN", "POUTCOME", "EDUCATION"],
    "df": [int(t3r.loc[c, "df"]) for c in ["contact", "age", "house_loan", "poutcome", "education"]],
    "Chi-kwadrat Walda (brak Y)": [f(t3r.loc[c, "chi2 Walda"]) for c in ["contact", "age", "house_loan", "poutcome", "education"]],
    "p (brak Y)": [fp(t3r.loc[c, "p"]) for c in ["contact", "age", "house_loan", "poutcome", "education"]],
    "Chi-kwadrat Walda (Y)": [f(t3y.loc[c, "chi2 Walda"]) for c in ["contact", "age", "house_loan", "poutcome", "education"]],
    "p (Y)": [fp(t3y.loc[c, "p"]) for c in ["contact", "age", "house_loan", "poutcome", "education"]]})
tabela(t, "Testy Walda dla cech (Type 3) w modelu prawdopodobieństwa obserwacji Y oraz w "
          "modelu Y (tylko obserwowane Y).")

akapit("Sprawdziliśmy też, czy cechy działają na brak Y tak samo w każdym kanale, dodając do "
       "modelu interakcje CONTACT × AGE, CONTACT × HOUSING_LOAN i CONTACT × EDUCATION. Interakcji "
       "CONTACT × POUTCOME nie badamy, bo w kanale 0 aż 99,3% klientów ma poutcome = unknown. "
       f"Istotna jest tylko interakcja CONTACT × AGE (p = {fp(dr.loc['CONTACT × AGE', 'p (brak Y)'])}). "
       f"W kanale B brak Y ma {pr(B['18-30'])} klientów w wieku 18–30 lat i tylko {pr(B['61+'])} "
       "klientów powyżej 60 lat, a w kanałach 0 i A odsetek braków zmienia się z wiekiem "
       "niewiele (rysunek 7). Mechanizm braku jest więc nieco inny w kanale B niż w pozostałych.")
rysunek("s2_w2_braki_wiek_kanal.png", "Odsetek braków Y według wieku i kanału komunikacji.")

# 3.
tytul("3. Związek cech z Y")
akapit("Aby ocenić, które cechy mogą pomóc w imputacji, oszacowaliśmy model logistyczny Y ze "
       "wszystkimi cechami na klientach z obserwowanym Y (n = 8 425). Model jest istotny "
       f"(chi-kwadrat = {f(gy['LR chi2'])}, df = {int(gy['df'])}, p < 0,001), a statystyka c "
       f"wynosi {f(gy['c (AUC)'], 3)}. Silnie z Y związane są CONTACT, HOUSING_LOAN i POUTCOME "
       "(tabela 6). EDUCATION, której nie ma w modelu docelowym, również jest istotna "
       f"(p = {fp(t3y.loc['education', 'p'])}): szansa zakupu klientów z wykształceniem "
       f"podstawowym jest o {f(100 * (1 - ory.loc['education[T.primary]', 'OR']), 0)}% mniejsza "
       f"niż klientów z wykształceniem średnim (OR = {f(ory.loc['education[T.primary]', 'OR'])}, "
       "tabela 7). Wiek w postaci liniowej nie jest istotny "
       f"(p = {fp(t3y.loc['age', 'p'])}), ale rysunek 8 pokazuje, że najczęściej kupują klienci "
       f"najmłodsi i najstarsi (w kanale B: {pr(BZ['18-30'])} w wieku 18–30 lat i "
       f"{pr(BZ['61+'])} powyżej 60 lat).")

NAZWY = {"contact[T.0]": "CONTACT: 0 wobec B", "contact[T.1]": "CONTACT: A wobec B",
         "house_loan[T.no]": "HOUSING_LOAN: no wobec yes", "poutcome[T.failure]": "POUTCOME: failure",
         "poutcome[T.success]": "POUTCOME: success", "poutcome[T.other]": "POUTCOME: other",
         "education[T.primary]": "EDUCATION: primary", "education[T.tertiary]": "EDUCATION: tertiary",
         "education[T.unknown]": "EDUCATION: unknown", "age": "AGE (1 rok)"}


def or_ci(t, k):
    return f"{f(t.loc[k, 'OR'], 3)} ({f(t.loc[k, 'dolna 95%'], 3)}–{f(t.loc[k, 'górna 95%'], 3)})"


t = pd.DataFrame({"Parametr": list(NAZWY.values()),
                  "OR obserwacji Y (95% PU)": [or_ci(orr, k) for k in NAZWY],
                  "OR zakupu (95% PU)": [or_ci(ory, k) for k in NAZWY]})
tabela(t, "Ilorazy szans w modelu prawdopodobieństwa obserwacji Y i w modelu Y. Poziomy "
          "odniesienia: kanał B, kredyt = yes, poutcome = unknown, education = secondary.")
akapit("W modelu Y istotna jest interakcja CONTACT × HOUSING_LOAN "
       f"(p = {fp(dr.loc['CONTACT × HOUSING_LOAN', 'p (Y)'])}), czyli różnica w szansie zakupu "
       "między klientami z kredytem i bez kredytu zależy od kanału. Interakcja CONTACT × AGE "
       f"nie jest istotna (p = {fp(dr.loc['CONTACT × AGE', 'p (Y)'])}), a CONTACT × EDUCATION "
       f"również nie (p = {fp(dr.loc['CONTACT × EDUCATION', 'p (Y)'])}).")
rysunek("s2_w3_zakup_wiek_kanal.png", "Odsetek zakupów (Y=1) według wieku i kanału komunikacji "
        "(tylko obserwowane Y).")

# 4.
tytul("4. Cechy pomocnicze w imputacji – tabela decyzyjna")
akapit("Tabela 8 zestawia związek każdej cechy i interakcji z brakiem Y i z Y (poziom istotności "
       "0,05) i przypisuje jej rolę według reguły: cecha związana z brakiem i z Y usuwa "
       "obciążenie, cecha związana tylko z Y poprawia precyzję, a cecha związana tylko z "
       "brakiem niewiele daje. Cechy modelu docelowego wchodzą do modelu imputacyjnego zawsze, "
       "żeby model imputacyjny zawierał wszystkie zmienne analizowanego modelu.")
t = dec.copy()
t["p (brak Y)"], t["p (Y)"] = t["p (brak Y)"].map(fp), t["p (Y)"].map(fp)
t = t[["element", "w modelu docelowym", "p (brak Y)", "p (Y)", "rola", "decyzja"]]
t.columns = ["Cecha", "W modelu docelowym", "p (brak Y)", "p (Y)", "Rola", "Decyzja"]
tabela(t, "Tabela decyzyjna: związek cech z brakiem Y i z Y oraz rola w modelu imputacyjnym.",
       num_od=2, num_do=4)
akapit("Żadna cecha nie jest związana jednocześnie z brakiem Y i z Y na poziomie 0,05. "
       "Najbliżej jest interakcja CONTACT × AGE, istotna dla braku i bliska istotności dla Y "
       f"(p = {fp(dr.loc['CONTACT × AGE', 'p (Y)'])}). EDUCATION i CONTACT × HOUSING_LOAN "
       "są związane tylko z Y, więc mogą poprawić precyzję imputacji, ale nie usuną obciążenia.")

# 5.
tytul("5. Projekt modelu imputacyjnego")
akapit("Brakuje tylko Y, więc wzorzec braków jest monotoniczny i wystarczy jeden model "
       "imputacyjny: regresja logistyczna dla zmiennej skokowej (w SAS: PROC MI z instrukcją "
       "MONOTONE LOGISTIC). Proponowaną specyfikację podaje tabela 9. Po imputacji model "
       "docelowy szacujemy w każdym ze 100 uzupełnionych zbiorów i łączymy wyniki regułami "
       "Rubina (PROC MIANALYZE). Przewidywane prawdopodobieństwa obserwacji Y z modelu z punktu "
       "2 zapisaliśmy w pliku s2_p_obserwacji.csv do metody wag w części 5.")
t = spec.copy()
t.columns = ["Element", "Propozycja"]
tabela(t, "Proponowana specyfikacja modelu imputacyjnego.", num_od=9)

# 6.
tytul("6. Wnioski")
akapit("Ani test t, ani testy chi-kwadrat, ani model logistyczny nie wykazały związku braku Y z "
       "pojedynczymi cechami, co według kryteriów z wykładu wskazuje na mechanizm MCAR. Jedynym "
       "odstępstwem jest zależność braku od wieku w kanale B. Wiek i kanał są w modelu "
       "docelowym, a zależność ta jest słaba (c = "
       f"{f(gr['c (AUC)'], 3)}), więc spodziewamy się, że wynik imputacji wielokrotnej będzie "
       "bliski analizie kompletnych przypadków. Imputacja może głównie poprawić precyzję dzięki "
       "cesze EDUCATION.")
akapit("Na podstawie próby nie można jednak stwierdzić, czy braki są typu MAR, czy MNAR. "
       "Wszystkie testy porównują cechy obserwowane, a nie mówią nic o tym, czy klienci bez Y "
       "kupowali częściej lub rzadziej niż podobni klienci z obserwowanym Y. Dlatego w części 5 "
       "przeprowadzamy analizę wrażliwości dla braków typu MNAR.")

tytul("Literatura")
akapit("Little, R. J. A., Rubin, D. B. (2002). Statistical Analysis with Missing Data (2nd ed.). "
       "Hoboken: John Wiley & Sons.")
akapit("Rubin, D. B. (1987). Multiple Imputation for Nonresponse in Surveys. New York: John "
       "Wiley & Sons.")

out = RAPORT / "02_raport_czesc2.docx"
doc.save(out)
print("Zapisano:", out)
