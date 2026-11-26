# =============================================================================
# Sekcja 5 raportu (diagnostyka imputacji, analiza wrażliwości, innowacja)
#   -> raport/04_raport_czesc5.docx
# Uruchamiać PO 04_diagnostyka.py i 04_recenzja_osoby1.py: liczby są czytane z plików
# s4_*.csv w wyniki/. Wymaga: python-docx.
# =============================================================================
import importlib
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

S = importlib.import_module("00_setup")
W = S.WYNIKI
RAPORT = S.BASE / "raport"
RAPORT.mkdir(exist_ok=True)
# Wygląd jak w częściach 1 i 2: dokument na bazie raportu części 1 (style, marginesy,
# czcionki), nagłówki zwykłym tekstem, podpisy nad tabelami i rysunkami.
# Numeracja jest kontynuacją wcześniejszych części: część 1 = tabele 1-3 i rysunki 1-5,
# część 2 = tabele 4-10 i rysunki 6-8, część 3 = 5 tabel (11-15) i bez rysunków.
SZABLON = RAPORT / "01_raport_czesc1.docx"
NR_TAB, NR_RYS = 16, 9                                    # pierwsze numery w tej części


def f(x, n=2):
    """Liczba w zapisie polskim (z minusem typograficznym; bez "−0,0")."""
    x = round(x, n) + 0.0
    return f"{x:.{n}f}".replace(".", ",").replace("-", "−")


def fp(p):
    return "< 0,001" if p < .001 else f(p, 3)


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
    """Nagłówek w stylu części 1 i 2: zwykły tekst w osobnym akapicie."""
    doc.add_paragraph(tekst)


def tabela(df, podpis, num_od=1, num_do=None):
    """Podpis nad tabelą, tabela z siatką; liczby wyrównane do prawej (jak w części 1 i 2)."""
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
    """Podpis nad rysunkiem, jak w części 1 i 2."""
    global NR_RYS
    doc.add_paragraph(f"Rysunek {NR_RYS}. {podpis}")
    NR_RYS += 1
    doc.add_picture(str(W / plik), width=Cm(szer))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def or_ci(w):
    return f"{f(w['OR A wobec B'])} ({f(w['OR dolna 95%'])}–{f(w['OR górna 95%'])})"


# --- dane z wyników 04_diagnostyka.py i 03_modele.py
ppc = pd.read_csv(W / "s4_t1_ppc.csv")
ppc.columns = ["model"] + list(ppc.columns[1:])
ppc_g = ppc[ppc.model.str.startswith("model z punktu")].set_index("statystyka")["PPC p"]
ppc_z = ppc[ppc.model.str.startswith("model celowo")].set_index("statystyka")
siatka = pd.read_csv(W / "s4_t2_mnar_siatka.csv")
przel = pd.read_csv(W / "s4_t4_punkt_przelamania.csv")
wagi = pd.read_csv(W / "s4_t5_ipw_wagi.csv")
war = pd.read_csv(W / "s4_t6_warianty_imputacji.csv")
ame = pd.read_csv(W / "s4_t7_efekt_prawdopodobienstwo.csv")
sym = pd.read_csv(W / "s4_t8_sztuczne_braki.csv")
wyn = pd.read_csv(W / "s4_tabela_wynikow.csv").set_index("metoda")
rec = pd.read_csv(W / "s4_recenzja_osoby1.csv")
imp = pd.read_csv(W / "s4_imputacje_glowny.csv")
s3 = pd.read_csv(W / "s3_tabela_wynikow.csv")
cc, mi3 = s3.iloc[0], s3.iloc[1]
mi5 = wyn.iloc[2]
M5 = len(imp)
B = imp["ocena"].var(ddof=1)
mc100, mc500 = np.sqrt(B / 100), np.sqrt(B / M5)
K = "beta1 - beta2 (A wobec B)"


def wiersz_przel(kier, strona):
    return przel[(przel.kierunek.str.startswith(kier)) & (przel.strona == strona)].iloc[0]


pA = wiersz_przel("tylko kanał A", "t > 0")
pBm = wiersz_przel("tylko kanał B", "t < 0")
pBp = wiersz_przel("tylko kanał B", "t > 0")
pPp = wiersz_przel("przeciwne", "t > 0")
pPm = wiersz_przel("przeciwne", "t < 0")

# =============================================================================
tytul("Część 5: diagnostyka imputacji i analiza wrażliwości")
akapit("W tej części odpowiadamy na punkt 5 zadania (miara diagnostyczna do oceny imputacji "
       "w kontekście analizowanego modelu) i sprawdzamy, jak mocny jest wniosek o różnicy między "
       "kanałami A i B. Wszystkie obliczenia są w pliku 04_diagnostyka.py. Model imputacyjny jest "
       "taki sam jak w części 3: regresja logistyczna Y względem CONTACT, AGE, HOUSING_LOAN, "
       "POUTCOME i EDUCATION z interakcją CONTACT × HOUSING_LOAN, z losowaniem parametrów.")

# --- 5.1 PPC
tytul("1. Proponowana miara: predykcyjna wartość p a posteriori dla β1 − β2")
akapit("Proponujemy predykcyjną wartość p a posteriori (ang. posterior predictive check, PPC; "
       "Nguyen, Lee i Carlin, 2015) liczoną dla tej samej wielkości, o którą pyta zadanie, "
       "czyli dla różnicy β1 − β2 w modelu docelowym. Procedura ma trzy kroki:")
for t in ["(1) W każdej imputacji, z tymi samymi wylosowanymi parametrami modelu imputacyjnego, "
          "losujemy Y także dla klientów z obserwowanym Y. Tak powstają dane replikowane: "
          "obserwowane Y zastąpione wartościami z modelu, imputowane Y bez zmian.",
          "(2) W każdej imputacji szacujemy model docelowy dwa razy: na danych uzupełnionych "
          "(obserwowane Y + imputowane) i na danych replikowanych.",
          "(3) PPC p to odsetek imputacji, w których β1 − β2 z danych replikowanych jest większe "
          "niż z danych uzupełnionych."]:
    akapit(t)
akapit("Jeśli model imputacyjny dobrze odtwarza zależność zakupu od kanału, oba wyniki różnią się "
       "tylko losowo i PPC p jest bliskie 0,5. Wartości bliskie 0 lub 1 oznaczają, że model "
       "imputacyjny systematycznie zawyża lub zaniża kontrast.")
akapit("Uzasadnienie wyboru. Po pierwsze, miara ocenia model imputacyjny dokładnie pod kątem "
       "wniosku, który z niego wyciągamy, a nie ogólnej jakości dopasowania. Po drugie, Y jest "
       "zmienną binarną, więc typowe narzędzia z literatury (porównanie gęstości wartości "
       "imputowanych i obserwowanych, wykresy paskowe) niewiele tu mówią: imputowane są same zera "
       "i jedynki. Po trzecie, przy MAR rozkład imputowanych Y może legalnie różnić się od "
       "obserwowanego, więc proste porównanie odsetków nie odróżnia dobrej imputacji od złej, "
       "a PPC porównuje dane z modelem przy tych samych cechach klientów.")

tab_ppc = pd.DataFrame({
    "Statystyka": ["β1 − β2 (A wobec B)", "odsetek Y = 1, kanał A", "odsetek Y = 1, kanał B",
                   "odsetek Y = 1, kanał 0"],
    "PPC p: model z punktu 2": [f(ppc_g[s], 3) for s in ppc_z.index],
    "PPC p: model celowo błędny (bez CONTACT)": [f(ppc_z.loc[s, "PPC p"], 3) for s in ppc_z.index]})
tabela(tab_ppc, f"Predykcyjne wartości p a posteriori ({M5} imputacji). Wartość bliska 0,5 "
                "oznacza dobre dopasowanie, bliska 0 lub 1 – złe.")
rysunek("s4_w1_ppc.png", "β1 − β2 na danych uzupełnionych i replikowanych w każdej imputacji.")
akapit(f"Dla modelu z punktu 2 PPC p dla kontrastu wynosi {f(ppc_g[K])}, a dla odsetków zakupów "
       f"w kanałach od {f(ppc_g.drop(K).min())} do {f(ppc_g.drop(K).max())}. Punkty na rysunku 9 "
       "leżą po obu stronach przekątnej. Model imputacyjny odtwarza więc zarówno kontrast A wobec B, "
       "jak i poziom zakupów w każdym kanale.")
akapit("Żeby sprawdzić, czy miara w ogóle potrafi wykryć zły model, powtórzyliśmy procedurę dla "
       "modelu celowo błędnego, bez zmiennej CONTACT. Odsetki zakupów w kanałach A i 0 są wtedy "
       f"wyraźnie niedopasowane (PPC p = {f(ppc_z.loc['odsetek Y=1, kanał A', 'PPC p'], 3)} i "
       f"{f(ppc_z.loc['odsetek Y=1, kanał 0', 'PPC p'], 3)}). Sam kontrast ma jednak PPC p = "
       f"{f(ppc_z.loc[K, 'PPC p'])}, czyli niezbyt skrajne, choć średni kontrast na danych "
       f"replikowanych ({f(ppc_z.loc[K, 'średnia replikowane'], 3)}) jest wielokrotnie mniejszy niż na "
       f"uzupełnionych ({f(ppc_z.loc[K, 'średnia uzupełnione'], 3)}). Powód: oszacowanie β1 − β2 ma "
       "duży błąd losowy, bo w kanale B jest tylko 546 obserwowanych Y.")
akapit("Ograniczenia miary. PPC p nie ma klasycznych progów typu 0,05 i nie jest wartością p "
       "testu: jest to opis zgodności modelu z danymi. Jego czułość spada, gdy braków jest dużo "
       "albo gdy statystyka jest mało precyzyjna, co pokazał model celowo błędny. Dlatego PPC dla "
       "kontrastu raportujemy razem z PPC dla odsetków zakupów w kanałach (te same imputacje, "
       "bez dodatkowych obliczeń) oraz z frakcją brakującej informacji FMI. FMI = "
       f"{f(mi5['FMI'])} mówi, ile informacji o kontraście utracono przez braki, ale nie mówi, czy "
       "model imputacyjny jest dobry, więc nie może być jedyną miarą.")

# --- 5.2 Stabilność MI
tytul("2. Błąd Monte Carlo imputacji")
akapit(f"Imputacja jest losowa, więc wynik zależy od liczby imputacji M. Przy M = 100 (część 3) "
       f"błąd Monte Carlo oceny β1 − β2 wynosi około {f(mc100, 4)}, a przy M = {M5} około "
       f"{f(mc500, 4)}. Powtórzenie imputacji tym samym modelem przy M = {M5} daje β1 − β2 = "
       f"{f(mi5['beta1 - beta2'], 3)} (OR = {or_ci(mi5)}), a w części 3 otrzymano "
       f"{f(mi3['beta1 - beta2'], 3)}. Różnica ({f(mi3['beta1 - beta2'] - mi5['beta1 - beta2'], 3)}) "
       "to kilka błędów Monte Carlo przy M = 100. Wynika z losowości imputacji, a nie z różnicy "
       "w modelu: współczynniki modelu imputacyjnego są w obu przebiegach identyczne. Wniosek: imputacja "
       f"wielokrotna daje praktycznie ten sam wynik co kompletne przypadki ({f(cc['beta1 - beta2'], 3)}). "
       f"Rekomendujemy raportowanie wyniku MI z M = {M5}.")

# --- 5.3 Warianty
tytul("3. Czy wynik zależy od modelu imputacyjnego i od metody korekty?")
tab_war = pd.DataFrame({
    "Wariant": war["wariant"],
    "β1 − β2": [f(v, 3) for v in war["ocena"]], "Błąd": [f(v, 3) for v in war["błąd"]],
    "OR A wobec B": [f(v) for v in war["OR A wobec B"]], "p": [fp(v) for v in war["p"]],
    "FMI": ["–" if pd.isna(v) else f(v) for v in war["FMI"]],
    "PPC p": ["–" if pd.isna(v) else f(v) for v in war["PPC p"]]})
tabela(tab_war, f"Porównanie modeli imputacyjnych (M = {M5}).")
akapit("Dodanie interakcji CONTACT × AGE, która w części 2 była istotna w modelu braku "
       "(p = 0,001), oraz usunięcie interakcji CONTACT × HOUSING_LOAN zmieniają ocenę β1 − β2 "
       f"o mniej niż {f(war['ocena'].iloc[:3].max() - war['ocena'].iloc[:3].min() + .005, 2)}, czyli "
       "o niewielki ułamek jej błędu standardowego. Wszystkie warianty mają PPC p bliskie 0,5. "
       "Wariant MID (imputacja, potem usunięcie imputowanych Y; von Hippel, 2007) jest w naszych "
       "danych dokładnie równy analizie kompletnych przypadków, bo brakuje wyłącznie Y. To "
       "potwierdza, że jedyną możliwą korektę w MI wnoszą EDUCATION i interakcje w modelu "
       "imputacyjnym.")
wa = wagi
akapit("Drugą metodą korekty przy MAR jest ważenie odwrotnością prawdopodobieństwa obserwacji Y "
       "(IPW). Wagi stabilizowane policzyliśmy z modelu braku z części 2 oraz z modelu rozszerzonego "
       "o CONTACT × AGE. Wagi są bliskie 1 (iloraz największej do najmniejszej "
       f"{f(wa['max/min'].iloc[0])} i {f(wa['max/min'].iloc[2])}), bo model braku prawie nie "
       "odróżnia klientów z brakiem Y od pozostałych (c = 0,517). Model docelowy z wagami i odpornymi "
       f"błędami daje OR od {f(wyn.filter(like='IPW', axis=0)['OR A wobec B'].min())} do "
       f"{f(wyn.filter(like='IPW', axis=0)['OR A wobec B'].max())}, czyli prawie dokładnie tyle co "
       "kompletne przypadki. Wniosek nie zależy więc od sposobu korekty przy MAR.")

# --- 5.4 MNAR
tytul("4. Analiza wrażliwości na braki nielosowe (MNAR)")
akapit("Dane nie pozwalają odróżnić MAR od MNAR: nie wiemy, czy klienci bez Y kupowali częściej "
       "czy rzadziej niż podobni klienci z Y. Dlatego sprawdzamy, jak bardzo musiałoby się różnić "
       "ich zachowanie, żeby zmienił się wniosek (metoda delta-adjustment; Carpenter i Kenward, "
       "2013). W imputacji dodajemy do logitu szansy zakupu klientów bez Y przesunięcie δA w kanale "
       "A i δB w kanale B (w grupie bez komunikacji δ = 0). exp(δ) mówi, ile razy szansa zakupu "
       "klienta bez Y jest większa od szansy podobnego klienta z Y; δ = 0 to MAR. Dla każdego "
       "punktu siatki od −2 do 2 co 0,5 wykonujemy pełną imputację i łączymy wyniki regułami "
       "Rubina, a punkt przełamania wyznaczamy na gęstszych liniach (krok 0,1, M = 100).")
rysunek("s4_w2_mnar_mapa.png", "OR zakupu A wobec B przy różnych założeniach o klientach bez Y.", szer=14)
rysunek("s4_w3_mnar_linie.png", "OR A wobec B wzdłuż trzech kierunków odchylenia od MAR; "
        "czerwone linie: punkty przełamania.")
tab_p = pd.DataFrame({
    "Kierunek odchylenia od MAR": przel["kierunek"] + ", " + przel["strona"],
    "δA": ["–" if pd.isna(v) else f(v, 1) for v in przel["deltaA"]],
    "δB": ["–" if pd.isna(v) else f(v, 1) for v in przel["deltaB"]],
    "exp(δA)": ["–" if pd.isna(v) else f(v) for v in przel["exp(deltaA)"]],
    "exp(δB)": ["–" if pd.isna(v) else f(v) for v in przel["exp(deltaB)"]],
    "Wniosek po przełamaniu": przel["wniosek po przełamaniu"]})
tabela(tab_p, "Punkty przełamania: najmniejsze odchylenie od MAR, przy którym 95% przedział "
              "ufności OR A wobec B przestaje zawierać 1.", num_do=5)
akapit("Wniosek z części 3 („brak istotnej różnicy między kanałami”) jest wrażliwy na MNAR "
       f"w jednym kierunku. Wystarczy, że klienci bez Y w kanale A mają szansę zakupu "
       f"{f(pA['exp(deltaA)'])} raza większą niż podobni klienci z Y (δA = {f(pA['deltaA'], 1)}), "
       f"albo że klienci bez Y w kanale B mają ją {f(1 / pBm['exp(deltaB)'])} raza mniejszą "
       f"(δB = {f(pBm['deltaB'], 1)}), a kanał A staje się istotnie lepszy. Przy przesunięciach "
       f"w przeciwnych kierunkach wystarczy δA = {f(pPp['deltaA'], 1)} i δB = {f(pPp['deltaB'], 1)} "
       f"(szanse {f(pPp['exp(deltaA)'])} raza i {f(pPp['exp(deltaB)'])} raza). Odwrotny wniosek, "
       "że kanał B jest istotnie lepszy, wymaga dużo silniejszego MNAR: "
       f"δB = {f(pBp['deltaB'], 1)} (szansa {f(pBp['exp(deltaB)'], 1)} raza większa) albo "
       f"δA = {f(pPm['deltaA'], 1)} i δB = {f(pPm['deltaB'], 1)}. Jednakowe przesunięcie w obu kanałach "
       "(przekątna mapy) prawie nie zmienia OR, bo działa na A i B tak samo.")
akapit("Czy takie odchylenia są wiarygodne? Status klienta jest nieznany w t1, więc brak Y może "
       "wynikać np. z braku kontaktu po kampanii albo z niedokończonego procesu zakupu. Różnica "
       "szans rzędu 1,3–1,5 raza między klientami bez Y i z Y nie jest nieprawdopodobna, natomiast "
       "różnica rzędu 2,7–4 razy, potrzebna do odwrócenia wniosku na korzyść kanału B, jest mało "
       "realistyczna. W praktyce oznacza to, że dane nie dają podstaw, by uznać kanał B za lepszy, "
       "a ocena „A i B nie różnią się” jest wnioskiem ostrożnym, który przy umiarkowanym MNAR "
       "przesunąłby się na korzyść kanału A.")

# --- 5.5 Efekt w punktach procentowych
tytul("5. Efekt na skali prawdopodobieństwa")
a_cc, a_mi = ame.iloc[0], ame.iloc[1]
akapit("Iloraz szans jest trudny do przełożenia na decyzję biznesową, dlatego policzyliśmy też "
       "średni efekt krańcowy: średnią po wszystkich klientach różnicę przewidywanego "
       "prawdopodobieństwa zakupu przy kanale A i przy kanale B, przy ich własnych pozostałych "
       "cechach (błąd metodą delta, w MI łączony regułami Rubina). Kanał A zwiększa "
       f"prawdopodobieństwo zakupu o {f(a_mi['AME (pp)'], 1)} punktu procentowego "
       f"(95% PU: {f(a_mi['dolna 95% (pp)'], 1)} do {f(a_mi['górna 95% (pp)'], 1)} pp, p = {fp(a_mi['p'])}) "
       f"przy MAR i o {f(a_cc['AME (pp)'], 1)} pp ({f(a_cc['dolna 95% (pp)'], 1)} do "
       f"{f(a_cc['górna 95% (pp)'], 1)} pp) w analizie kompletnych przypadków. Przy bazowym poziomie "
       "zakupów ok. 13–15% to niewielka i niepewna różnica: przedział obejmuje zarówno brak efektu, "
       "jak i przewagę kanału A o ponad 4 pp.")

# --- 5.6 Sztuczne braki
tytul("6. Test na sztucznych brakach")
prawda = sym["wartość prawdziwa"].iloc[0]
akapit("Na koniec sprawdziliśmy, czy przyjęta metoda odzyskuje znaną wartość. Zachowaliśmy cechy "
       "wszystkich 11 304 klientów, a Y losowaliśmy z modelu dopasowanego do obserwowanych danych, "
       "celowo bogatszego niż model imputacyjny (wiek w przedziałach i jego interakcja z kanałem). "
       f"Wartość prawdziwa β1 − β2 w tym modelu wynosi {f(prawda, 3)}. Następnie usuwaliśmy część Y "
       "jednym z dwóch mechanizmów MAR: (1) jak w naszych danych, przez interakcję CONTACT × AGE; "
       "(2) silnie przez wykształcenie, zwłaszcza w kanale B. Każdy scenariusz powtórzyliśmy "
       "200 razy (MI z 10 imputacjami).")
tab_s = pd.DataFrame({
    "Mechanizm": sym["mechanizm"].str.split(":").str[0], "Metoda": sym["metoda"],
    "% braków": [f(100 * v, 1) for v in sym["odsetek_brakow"]],
    "Obciążenie": [f(v, 3) for v in sym["obciążenie"]],
    "Błąd MC obciążenia": [f(v, 3) for v in sym["błąd MC obciążenia"]],
    "RMSE": [f(v, 3) for v in sym["RMSE"]],
    "Pokrycie 95% PU": [f(100 * v, 1) + "%" for v in sym["pokrycie_95"]]})
tabela(tab_s, "Test na sztucznych brakach: 200 powtórzeń w każdym scenariuszu.", num_od=2)
akapit("W obu scenariuszach obie metody mają obciążenie małe w porównaniu z błędem standardowym "
       f"kontrastu (ok. 0,14) i pokrycie przedziałów bliskie 95% (od {f(100 * sym['pokrycie_95'].min(), 1)}% "
       f"do {f(100 * sym['pokrycie_95'].max(), 1)}%). Imputacja nie poprawia tu wyniku, bo w modelu "
       "docelowym są już cechy najsilniej związane z zakupem, a słaby mechanizm braku nie "
       "wprowadza istotnego obciążenia. Test potwierdza więc obraz z części 2 i 3: przy MAR wynik "
       "dla kontrastu A wobec B jest wiarygodny niezależnie od metody, a główną niepewnością jest "
       "założenie MAR, które bada analiza wrażliwości.")

# --- 5.7 Wnioski
tytul("7. Wnioski")
for t in [f"Miarą diagnostyczną jest PPC p dla β1 − β2. Dla przyjętego modelu imputacyjnego wynosi "
          f"{f(ppc_g[K])}, a dla odsetków zakupów w kanałach jest bliskie 0,5: model imputacyjny jest "
          "dobrze dopasowany do wielkości, o którą pyta zadanie.",
          "Wynik przy MAR jest stabilny: imputacja wielokrotna w kilku wariantach, MID i IPW dają "
          f"OR A wobec B od {f(wyn['OR A wobec B'].drop(wyn.filter(like='MNAR', axis=0).index).min())} do "
          f"{f(wyn['OR A wobec B'].drop(wyn.filter(like='MNAR', axis=0).index).max())}, zawsze z "
          "przedziałem ufności zawierającym 1.",
          "Wniosek o braku różnicy jest wrażliwy na umiarkowane MNAR na korzyść kanału A "
          f"(exp(δ) ≈ {f(pPp['exp(deltaA)'], 1)}–{f(pA['exp(deltaA)'], 1)}), natomiast przewaga "
          "kanału B wymagałaby nierealistycznie silnego MNAR.",
          f"Na skali biznesowej kanał A daje ok. {f(a_mi['AME (pp)'], 1)} pp więcej zakupów niż B, "
          "z dużą niepewnością. Nie ma podstaw, by preferować kanał B."]:
    akapit(t)

# --- recenzja
tytul("Recenzja krzyżowa: tabele z części 1")
akapit(f"Osoba 4 sprawdziła {len(rec)} liczb z tabel części 1, licząc je od nowa z danych "
       f"(plik 04_recenzja_osoby1.py). Niezgodności: {(rec['zgodne'] == 'NIE').sum()}. Liczności "
       "wszystkich tabel sumują się do 11 304 klientów, a tabel na obserwowanych Y do 8 425.")

tytul("Literatura")
for t in ["Carpenter, J. R., Kenward, M. G. (2013). Multiple Imputation and its Application. "
          "Chichester: Wiley.",
          "Nguyen, C. D., Lee, K. J., Carlin, J. B. (2015). Posterior predictive checking of multiple "
          "imputation models. Biometrical Journal, 57(4), 676–694.",
          "Nguyen, C. D., Carlin, J. B., Lee, K. J. (2017). Model checking in multiple imputation: "
          "an overview and case study. Emerging Themes in Epidemiology, 14, 8.",
          "von Hippel, P. T. (2007). Regression with missing Ys: an improved strategy for analyzing "
          "multiply imputed data. Sociological Methodology, 37(1), 83–117.",
          "von Hippel, P. T. (2020). How many imputations do you need? A two-stage calculation using "
          "a quadratic rule. Sociological Methods & Research, 49(3), 699–718."]:
    akapit(t)

out = RAPORT / "04_raport_czesc5.docx"
doc.save(out)
print("Zapisano:", out)
