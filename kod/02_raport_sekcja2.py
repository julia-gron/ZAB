# =============================================================================
# Sekcja 2 raportu (mechanizm braków) -> raport/02_raport_czesc2.docx
# Uruchamiać PO 02_mechanizm.py: tabele są czytane z plików s2_*.csv w wyniki/,
# więc liczby w tabelach raportu zgadzają się z wydrukiem. Wymaga: python-docx.
# =============================================================================
import importlib
import pandas as pd
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

S = importlib.import_module("00_setup")
W = S.WYNIKI
RAPORT = S.BASE / "raport"
RAPORT.mkdir(exist_ok=True)


def f(x, n=2):
    """Liczba w zapisie polskim; p < 0,001 zapisywane jako '< 0,001'."""
    if isinstance(x, str):
        return x
    return f"{x:.{n}f}".replace(".", ",")


def fp(p):
    return "< 0,001" if p < .001 else f(p, 3)


doc = Document()
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(11)
licznik = {"tab": 0, "rys": 0}


def akapit(t):
    doc.add_paragraph(t).paragraph_format.space_after = Pt(6)


def podpis(rodzaj, t):
    licznik[rodzaj] += 1
    nazwa = "Tabela" if rodzaj == "tab" else "Rysunek"
    p = doc.add_paragraph()
    r = p.add_run(f"{nazwa} 2.{licznik[rodzaj]}. ")
    r.bold = True
    p.add_run(t).italic = True


def tabela(df, tytul, szer=None):
    podpis("tab", tytul)
    t = doc.add_table(rows=1, cols=len(df.columns), style="Light Grid Accent 1")
    for i, c in enumerate(df.columns):
        t.rows[0].cells[i].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    doc.add_paragraph()


def rysunek(plik, tytul, szer=16):
    doc.add_picture(str(W / plik), width=Cm(szer))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    podpis("rys", tytul)


# --- dane z wyników 02_mechanizm.py
kat = pd.read_csv(W / "s2_t1_braki_wg_kategorii.csv")
chi = pd.read_csv(W / "s2_t2_chi2_braki.csv", index_col=0)
orr = pd.read_csv(W / "s2_t4_model_braku_or.csv", index_col=0)
t3r = pd.read_csv(W / "s2_t5_model_braku_type3.csv", index_col=0)
testy_r = pd.read_csv(W / "s2_t6_model_braku_testy.csv", index_col=0)
wiek_r = pd.read_csv(W / "s2_t6a_wiek_braki_w_kanalach.csv", index_col=0)
wiek_y = pd.read_csv(W / "s2_t7a_wiek_y_w_kanalach.csv", index_col=0)
testy_y = pd.read_csv(W / "s2_t8_model_y_testy.csv", index_col=0)
kand = pd.read_csv(W / "s2_t9_kandydaci_modelu_imputacyjnego.csv", index_col=0)
dec = pd.read_csv(W / "s2_t10_tabela_decyzyjna.csv")
spec = pd.read_csv(W / "s2_t11_projekt_modelu_imputacyjnego.csv")

doc.add_heading("Część 2: mechanizm braków danych", level=1)
akapit("Pytanie z punktu 2 zadania brzmi: czy prawdopodobieństwo braku Y_NEWDEPOSIT zależy od "
       "pozostałych cech i czy istnieją cechy, za pomocą których imputacja może skorygować "
       "oszacowania. Braki występują wyłącznie w Y (2 879 z 11 304 klientów, 25,47%), więc "
       "mechanizm opisujemy wskaźnikiem r (r = 1: brak Y). Cecha nadaje się do korekty, gdy wiąże "
       "się jednocześnie z r i z Y. Związek z r badamy na całej próbie, związek z Y na "
       "klientach z obserwowanym Y (n = 8 425).")

# 2.1
doc.add_heading("2.1. Odsetek braków w kategoriach cech", level=2)
akapit("Odsetek braków Y jest bliski średniej 25,5% we wszystkich kategoriach wszystkich cech "
       "(rysunek 2.1). Żaden test chi-kwadrat niezależności nie odrzuca hipotezy o braku związku, "
       "a V Cramera nie przekracza 0,025 (tabela 2.1). Największe odchylenie dotyczy kategorii "
       "education = unknown (29,5% braków, n = 492), ale całej cechy test nie wskazuje (p = "
       f"{fp(chi.loc['education', 'p'])}).")
t = chi.reset_index()
t.columns = ["cecha", "chi2", "df", "p", "V Cramera"]
t["cecha"] = t["cecha"].map({"contact": "CONTACT", "house_loan": "HOUSING_LOAN",
                             "education": "EDUCATION", "poutcome": "POUTCOME",
                             "age_q": "AGE (kwintyle)"})
t["chi2"], t["df"], t["p"], t["V Cramera"] = (t.chi2.map(f), t.df.astype(int), t.p.map(fp),
                                               t["V Cramera"].map(lambda x: f(x, 3)))
tabela(t, "Testy niezależności r od cech (chi-kwadrat).")
rysunek("s2_w1_braki_wg_cech.png", "Odsetek braków Y w kategoriach cech, z 95% przedziałami Wilsona.")
akapit("Wiek klientów z brakiem Y i z obserwowanym Y ma niemal identyczny rozkład: średnie 40,9 "
       "i 41,0 roku (SMD = 0,01), test t Welcha p = 0,692, test Manna-Whitneya p = 0,922, "
       "test Kołmogorowa-Smirnowa p = 0,892 (rysunek 2.2).")
rysunek("s2_w2_age_wg_r.png", "Rozkład wieku według r: histogram i dystrybuanty empiryczne.")

# 2.2
doc.add_heading("2.2. Model prawdopodobieństwa braku", level=2)
lr_all = testy_r.loc["wszystkie cechy łącznie"]
akapit("Model logistyczny r ~ CONTACT + AGE + HOUSING_LOAN + POUTCOME + EDUCATION potwierdza "
       "obraz z tabel. Łączny test ilorazu wiarygodności wszystkich cech daje "
       f"chi2 = {f(lr_all['LR chi2'])} przy {int(lr_all['df'])} stopniach swobody (p = "
       f"{fp(lr_all['p'])}), żadna cecha nie jest istotna w teście Type 3 (tabela 2.2), "
       "a AUC wynosi 0,517 (95% przedział bootstrap: 0,505–0,529). Model prawie nie odróżnia "
       "klientów z brakiem Y od pozostałych. Jedyny parametr istotny na poziomie 0,05 "
       "(education = unknown, OR = 1,25) nie przekłada się na istotność całej cechy. "
       "Wariant z AGE jako B-splajnem (4 st. sw.) nie poprawia modelu (p = 0,313, wyższe AIC).")
t = orr.reset_index()
t.columns = ["parametr", "beta", "SE", "OR", "dolna 95%", "górna 95%", "p"]
t = t[["parametr", "OR", "dolna 95%", "górna 95%", "p"]]
t["parametr"] = t["parametr"].map({
    "contact[T.0]": "CONTACT 0 wobec B", "contact[T.1]": "CONTACT A wobec B",
    "house_loan[T.no]": "HOUSING_LOAN no wobec yes", "poutcome[T.failure]": "POUTCOME failure",
    "poutcome[T.success]": "POUTCOME success", "poutcome[T.other]": "POUTCOME other",
    "education[T.primary]": "EDUCATION primary", "education[T.tertiary]": "EDUCATION tertiary",
    "education[T.unknown]": "EDUCATION unknown", "age": "AGE (1 rok)"})
for c in ["OR", "dolna 95%", "górna 95%"]:
    t[c] = t[c].map(lambda x: f(x, 3))
t["p"] = t["p"].map(fp)
tabela(t, "Model braku: ilorazy szans braku Y (OR > 1: większa szansa braku). Poziomy "
          "odniesienia: kanał B, kredyt = yes, poutcome = unknown, education = secondary; "
          "AGE: OR na 1 rok.")
akapit("Uzupełniająco policzono test MCAR Little’a (1988): d2 = 12,74, df = 10, p = 0,239. "
       "Test zakłada wielowymiarowy rozkład normalny, którego zmienne 0/1 nie spełniają, i bada "
       "tylko różnice średnich, więc traktujemy go jako potwierdzenie, a nie rozstrzygnięcie.")

# 2.3
doc.add_heading("2.3. Mechanizm braku różni się między kanałami", level=2)
ca = testy_r.loc["interakcje: contact x age"]
akapit("Model z samymi efektami głównymi zakłada, że każda cecha działa na brak Y tak samo w "
       "każdym kanale. Łączny test interakcji CONTACT × cecha daje p = "
       f"{fp(testy_r.loc['interakcje: wszystkie łącznie', 'p'])}, a za tę zależność odpowiada "
       f"interakcja CONTACT × AGE (chi2 = {f(ca['LR chi2'])}, df = 2, p = {f(ca['p'], 4)}). "
       "Wynik utrzymuje się po poprawce Bonferroniego na kilkanaście testów wykonanych w tej "
       "sekcji (próg 0,003). Interakcję CONTACT × POUTCOME testowano tylko dla kanału A wobec B, "
       "bo w kanale 0 jedynie 23 klientów ma poutcome inne niż unknown.")
t = wiek_r.reset_index()
t.columns = ["kanał", "n", "OR braku na 10 lat", "dolna 95%", "górna 95%", "p"]
t2 = wiek_y.reset_index()
t["OR zakupu na 10 lat (obserwowane Y)"] = t2["OR zakupu na 10 lat"].map(lambda x: f(x, 3))
t["p (zakup)"] = t2["p"].map(fp)
for c in ["OR braku na 10 lat", "dolna 95%", "górna 95%"]:
    t[c] = t[c].map(lambda x: f(x, 3))
t["p"] = t["p"].map(fp)
tabela(t, "Efekt wieku (na 10 lat) na szansę braku Y i na szansę zakupu, osobno w każdym kanale "
          "(modele z samym wiekiem).")
akapit("W kanale B szansa braku Y spada z wiekiem (OR = 0,82 na 10 lat, p = 0,001): brak Y ma "
       "33% klientów w wieku 18–30 lat i 18% klientów powyżej 60 lat (rysunek 2.3). W kanale 0 "
       "zależność jest odwrotna i słabsza (OR = 1,10, p = 0,027), w kanale A jej nie ma. "
       "Jednocześnie zależność szansy zakupu od wieku ma kształt litery U we wszystkich kanałach: "
       "najczęściej kupują klienci najmłodsi i powyżej 60 lat (w kanale B: 22% w wieku 18–30, "
       "6–10% w wieku 31–60, 34% powyżej 60 lat). Wśród obserwowanych Y kanału B najmłodsi są "
       "więc niedoreprezentowani, a najstarsi nadreprezentowani, a obie te grupy kupują znacznie "
       "częściej niż klienci w średnim wieku. Kierunek ewentualnego obciążenia nie jest "
       "oczywisty i zależy od tego, jak model docelowy (wiek liniowo) uśrednia ten kształt. Wiek jest zatem "
       "związany zarówno z r, jak i z Y, choć nie widać tego w modelu z efektami głównymi i "
       "liniowym wiekiem. Siła tego mechanizmu jest jednak mała: AUC modelu braku rośnie "
       "z 0,517 do 0,526.")
rysunek("s2_w4_wiek_kanal_braki_zakup.png",
        "Odsetek braków Y (lewy panel) i odsetek zakupów wśród obserwowanych Y (prawy panel) "
        "według wieku i kanału, z 95% przedziałami Wilsona.")

# 2.4
doc.add_heading("2.4. Związek cech z Y", level=2)
edu = testy_y.loc["EDUCATION ponad model docelowy"]
bs = testy_y.loc["nieliniowość AGE (B-splajn)"]
akapit("Model logistyczny Y na obserwowanych Y ze wszystkimi cechami (AUC = 0,712) pokazuje "
       "silny związek Y z CONTACT, HOUSING_LOAN i POUTCOME (tabela 2.4, rysunek 2.4). "
       f"EDUCATION wnosi informację o Y ponad cechy modelu docelowego (test LR: "
       f"chi2 = {f(edu['LR chi2'])}, df = 3, p = {fp(edu['p'])}); szansę zakupu obniża głównie "
       "wykształcenie podstawowe (OR = 0,71 wobec średniego). Przyrost AUC jest jednak mały "
       "(0,711 → 0,712). AGE w postaci liniowej jest nieistotny (p = 0,422), ale B-splajn "
       f"poprawia model bardzo wyraźnie (chi2 = {f(bs['LR chi2'])}, df = 3, p < 0,001; AIC spada "
       "o 112), co odpowiada kształtowi U z rysunku 2.3.")
akapit("Przy liniowym wieku istotne są interakcje CONTACT × AGE (p = 0,005) i CONTACT × "
       "HOUSING_LOAN (p < 0,001). Po zastąpieniu wieku splajnem interakcja z wiekiem słabnie "
       f"(p = {fp(testy_y.loc['interakcje (AGE splajn): contact x age', 'p'])}), bo częściowo "
       "wynikała z nieliniowości połączonej ze starszym składem kanału B. Interakcja z kredytem "
       f"pozostaje (p = {fp(testy_y.loc['interakcje (AGE splajn): contact x house_loan', 'p'])}). "
       "Interakcje z POUTCOME i EDUCATION są nieistotne.")
tt = testy_y.reset_index()
tt.columns = ["test", "LR chi2", "df", "p"]
tt["LR chi2"], tt["df"], tt["p"] = tt["LR chi2"].map(f), tt["df"].astype(int), tt["p"].map(fp)
tabela(tt, "Model Y (obserwowane Y): testy ilorazu wiarygodności.")
rysunek("s2_w3_or_braku_i_y.png", "Ilorazy szans tych samych cech w modelu braku i w modelu Y "
        "(wspólna skala). AGE: OR na 10 lat.")

# 2.5
doc.add_heading("2.5. Tabela decyzyjna", level=2)
akapit("Tabela 2.5 zestawia związek każdego elementu z r i z Y i przypisuje mu rolę według "
       "reguły: element związany z r i z Y usuwa obciążenie, związany tylko z Y poprawia "
       "precyzję, związany tylko z r niewiele daje. Cechy modelu docelowego wchodzą do modelu "
       "imputacyjnego niezależnie od wyniku, bo model imputacyjny musi zawierać wszystkie "
       "zmienne modelu analizy (White, Royston i Wood, 2011). Interakcję CONTACT × AGE "
       "włączamy mimo p(Y) = 0,080: jest silnie związana z r, a obciążenie zależy od iloczynu "
       "obu związków. Zbędny predyktor w modelu imputacyjnym kosztuje niewiele precyzji, "
       "pominięty potrzebny daje obciążenie (Collins, Schafer i Kam, 2001).")
t = dec.copy()
t["p (r)"], t["p (Y)"] = t["p (r)"].map(fp), t["p (Y)"].map(fp)
t = t[["element", "w modelu docelowym", "p (r)", "p (Y)", "rola wg reguły", "decyzja"]]
tabela(t, "Tabela decyzyjna: związek z r (model braku, test LR), związek z Y (model Y na "
          "obserwowanych Y, test LR; interakcje przy AGE jako splajnie), rola w modelu "
          "imputacyjnym. Poziom istotności 0,05.")

# 2.6
doc.add_heading("2.6. Projekt modelu imputacyjnego", level=2)
akapit("Specyfikację wybrano, porównując kryterium AIC dla siedmiu kandydatów na model Y "
       "(tabela 2.6). Wszyscy kandydaci zawierają cechy modelu docelowego i EDUCATION. "
       "Najniższe AIC ma wariant K5: splajn wieku oraz interakcje CONTACT × AGE i CONTACT × "
       "HOUSING_LOAN (LR wobec K2: chi2 = 12,89, df = 4, p = 0,012). BIC wskazuje prostszy "
       "K2, różnica dotyczy tylko interakcji, dlatego K2 zostaje wariantem do analizy "
       "wrażliwości.")
t = kand.reset_index()
t["logL"], t["AIC"], t["BIC"] = t.logL.map(f), t.AIC.map(lambda x: f(x, 1)), t.BIC.map(lambda x: f(x, 1))
t["AUC"], t["delta AIC"] = t.AUC.map(lambda x: f(x, 3)), t["delta AIC"].map(lambda x: f(x, 1))
tabela(t, "Kandydaci na model imputacyjny Y (dopasowanie na obserwowanych Y).")
tabela(spec, "Proponowana specyfikacja modelu imputacyjnego (na bramkę).")
akapit("Formuła modelu imputacyjnego dla Y: y ~ contact + house_loan + poutcome + education + "
       "bs(age, df = 4) + contact × age + contact × house_loan. Wskaźnik r nie wchodzi do modelu. "
       "Ponieważ brakuje tylko Y, imputacja jest jednokrokowa: losowanie parametrów regresji "
       "logistycznej z ich rozkładu a posteriori i losowanie Y z przewidywanego "
       "prawdopodobieństwa. Liczbę imputacji ustala Osoba 3 po pilotażu na 20 imputacjach "
       "regułą von Hippela (2020). Przewidywane P(r = 0 | X) z modelu braku z interakcją "
       "CONTACT × AGE zapisano w pliku s2_p_obserwacji.csv (kolumna p_obs_int, zakres "
       "0,578–0,887) do wag IPW w części 4.")

# 2.7
doc.add_heading("2.7. Wnioski", level=2)
akapit("Dane nie dają podstaw do odrzucenia MCAR w prostej postaci: brak Y nie zależy od "
       "żadnej cechy z osobna, a model braku z efektami głównymi ma AUC 0,517. MCAR odrzuca "
       "dopiero interakcja CONTACT × AGE: w kanale B brak Y występuje częściej u młodszych "
       "klientów. Ponieważ szansa zakupu zależy od wieku nieliniowo, a model docelowy zawiera "
       "wiek tylko liniowo i bez interakcji z kanałem, analiza kompletnych przypadków może być "
       "lekko obciążona. Imputacja z modelem K5 tę zależność uwzględnia. Mechanizm jest słaby, "
       "więc oczekujemy, że wynik MI będzie bliski analizie kompletnych przypadków. Taki wynik "
       "nie będzie błędem, tylko konsekwencją słabego związku cech z r. EDUCATION jest "
       "związana z Y, ale nie z r, więc może poprawić precyzję, a nie usunąć obciążenie.")
akapit("Dane mogą obalić MCAR, ale nie odróżnią MAR od MNAR. Wszystkie testy z tej sekcji "
       "porównują obserwowane cechy klientów z brakiem Y i bez niego. Nie mówią nic o tym, czy "
       "klienci bez Y kupowali częściej lub rzadziej niż podobni klienci z obserwowanym Y. "
       "Założenie MAR jest więc nietestowalne, co uzasadnia analizę wrażliwości MNAR w części 5.")

doc.add_heading("Literatura", level=2)
for z in ["Collins, L. M., Schafer, J. L., Kam, C.-M. (2001). A comparison of inclusive and "
          "restrictive strategies in modern missing data procedures. Psychological Methods, 6(4), "
          "330–351.",
          "Little, R. J. A. (1988). A test of missing completely at random for multivariate data "
          "with missing values. Journal of the American Statistical Association, 83(404), 1198–1202.",
          "von Hippel, P. T. (2020). How many imputations do you need? A two-stage calculation "
          "using a quadratic rule. Sociological Methods & Research, 49(3), 699–718.",
          "White, I. R., Royston, P., Wood, A. M. (2011). Multiple imputation using chained "
          "equations: Issues and guidance for practice. Statistics in Medicine, 30(4), 377–399."]:
    akapit(z)

out = RAPORT / "02_raport_czesc2.docx"
doc.save(out)
print("Zapisano:", out)
