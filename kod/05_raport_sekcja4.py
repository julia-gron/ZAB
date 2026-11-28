# =============================================================================
# Część 4 raportu (podsumowanie wyników w kontekście braków) -> raport/05_raport_czesc4.docx
# Uruchamiać PO 05_podsumowanie.py i 04_recenzja_osoby1.py. Liczby są czytane z plików
# wynikowych, więc raport zgadza się z wydrukiem. Wygląd jak w części 1 (raport_styl.py).
# Numeracja: część 3 kończy się na tabeli 15, części 1-3 mają rysunki 1-8.
# =============================================================================
import pandas as pd
from raport_styl import Raport, f, fp, pr, S

W = S.WYNIKI
R = Raport(nr_tab=16, nr_rys=9)

t = pd.read_csv(W / "s5_tabela_zbiorcza.csv")
cc, mi = t.iloc[0], t.iloc[1]
mar = t[t["założenie"] == "MAR"]
mnar = t[t["założenie"] == "MNAR"]
przel = pd.read_csv(W / "s4_t6_punkt_przelamania.csv")
glob = pd.read_csv(W / "s2_t10_testy_globalne.csv", index_col=0)
y_kanal = pd.read_csv(W / "t4_y_wg_contact.csv", index_col=0)
rec = pd.read_csv(W / "s4_recenzja_osoby1.csv")
ppc = pd.read_csv(W / "s4_t2_ppc.csv")


def przelamanie(kierunek, strona):
    w = przel[(przel["kierunek"] == kierunek) & (przel["strona"] == strona)].iloc[0]
    return w


pA = przelamanie("tylko kanał A (δB = 0)", "delta > 0")
pBm = przelamanie("tylko kanał B (δA = 0)", "delta < 0")
pBp = przelamanie("tylko kanał B (δA = 0)", "delta > 0")
c_braku = glob.loc["model braku (P obserwacji Y)", "c (AUC)"]

R.tytul("Część 4: podsumowanie wyników w kontekście braków danych")
R.akapit("W tej części zestawiamy wyniki estymacji modelu docelowego wszystkimi metodami "
         "użytymi w pracy (punkt 4 zadania) i odpowiadamy na trzy pytania: czy kanały A i B "
         "różnią się szansą zakupu, czy uwzględnienie braków danych zmienia wniosek oraz jakie "
         "założenie musi być prawdziwe, żeby wniosek się utrzymał.")

R.tytul("1. Wyniki wszystkich metod")
tab = pd.DataFrame({
    "Metoda": t["metoda"],
    "Założenie": t["założenie"],
    "β1 − β2": [f(x, 3) for x in t["beta1 - beta2"]],
    "Błąd": [f(x, 3) for x in t["błąd"]],
    "OR A wobec B (95% PU)": [f"{f(o)} ({f(a)}–{f(b)})" for o, a, b in
                              zip(t["OR A wobec B"], t["OR dolna 95%"], t["OR górna 95%"])],
    "p": [fp(x) for x in t["p"]],
    "FMI": ["–" if pd.isna(x) else f(x) for x in t["FMI"]]})
nr_tab = R.tabela(tab, "Iloraz szans zakupu w kanale A wobec kanału B według metody "
                       "uwzględnienia braków danych.", num_od=2)
nr_rys = R.rysunek("s5_w1_wykres_lesny.png", "Iloraz szans zakupu A wobec B według metody, "
                   "z 95% przedziałami ufności.")
R.akapit(f"Wszystkie metody oparte na założeniu MCAR lub MAR dają niemal ten sam wynik: OR A "
         f"wobec B od {f(mar['OR A wobec B'].min())} do {f(max(cc['OR A wobec B'], mar['OR A wobec B'].max()))}, "
         f"zawsze z przedziałem ufności zawierającym 1 (tabela {nr_tab}, rysunek {nr_rys}). "
         "Wynik zmienia się dopiero przy założeniu, że braki są nielosowe (MNAR).")

R.tytul("2. Czy kanały A i B różnią się szansą zakupu?")
R.akapit(f"Przyjmujemy jako wynik główny imputację wielokrotną przy założeniu MAR. Szansa zakupu "
         f"w kanale A jest {f(mi['OR A wobec B'])} raza większa niż w kanale B przy tych samych "
         f"pozostałych cechach klienta (95% PU: {f(mi['OR dolna 95%'])}–{f(mi['OR górna 95%'])}, "
         f"p = {fp(mi['p'])}). Przedział ufności zawiera 1, więc na poziomie istotności 0,05 nie "
         "ma podstaw, by twierdzić, że kanały różnią się szansą zakupu. Na obserwowanych danych "
         f"kupiło {pr(y_kanal.loc['A: kanał A', 'odsetek'])} klientów kanału A i "
         f"{pr(y_kanal.loc['B: kanał B', 'odsetek'])} klientów kanału B (tabela 3). Oba kanały "
         "wyraźnie przewyższają brak komunikacji, w której szansa zakupu jest około dwa razy "
         "mniejsza niż w kanale B (część 3).")

R.tytul("3. Czy uwzględnienie braków danych zmienia wniosek?")
R.akapit(f"Nie. Analiza kompletnych przypadków (OR = {f(cc['OR A wobec B'])}), imputacja "
         f"wielokrotna (OR = {f(mi['OR A wobec B'])}), jej warianty z innym modelem imputacyjnym "
         "oraz ważenie odwrotnością prawdopodobieństwa obserwacji (IPW) dają praktycznie tę samą "
         "ocenę i ten sam błąd standardowy. Przyczyny wynikają z części 2: brak Y prawie nie "
         f"zależy od cech klientów (statystyka c modelu braku = {f(c_braku, 3)}), a cechy silnie "
         "związane z zakupem są już w modelu docelowym. Jedyna cecha pomocnicza związana z "
         "zakupem, EDUCATION, nie wiąże się z brakiem Y, więc może poprawić precyzję, ale nie "
         f"usuwa obciążenia. Frakcja brakującej informacji FMI = {f(mi['FMI'])} oznacza, że braki "
         "odebrały około jednej czwartej informacji o różnicy między kanałami, czyli mniej więcej "
         "tyle, ile wynosi odsetek braków Y (25,5%). Diagnostyka z części 5 potwierdza, że model "
         f"imputacyjny odtwarza kontrast A wobec B (PPC p = {f(ppc.iloc[0]['PPC p'])}).")

R.tytul("4. Jakie założenie musi być prawdziwe?")
R.akapit("Wynik główny zakłada, że braki są typu MAR: klient bez Y kupuje z takim samym "
         "prawdopodobieństwem jak podobny klient z obserwowanym Y. Tego założenia nie da się "
         "sprawdzić na danych. Analiza wrażliwości w części 5 pokazuje, jak silne musiałoby być "
         "odstępstwo od MAR, żeby wniosek się zmienił. Kanał A stałby się istotnie lepszy, gdyby "
         f"klienci bez Y w kanale A mieli szansę zakupu {f(pA['exp(delta)'])} raza większą niż "
         f"podobni klienci z Y albo gdyby klienci bez Y w kanale B mieli ją "
         f"{f(1 / pBm['exp(delta)'])} raza mniejszą. Przewaga kanału B wymagałaby, żeby klienci "
         f"bez Y w kanale B kupowali z szansą aż {f(pBp['exp(delta)'])} raza większą niż podobni "
         "klienci z Y, co jest mało realistyczne.")

R.tytul("5. Wnioski i rekomendacja")
R.akapit(f"Dane nie pokazują istotnej różnicy w skuteczności kanałów A i B: szansa zakupu w "
         f"kanale A jest szacowana na {f(mi['OR A wobec B'])} raza większą niż w kanale B, ale "
         "z dużą niepewnością. Wniosek nie zależy od sposobu uwzględnienia braków danych. "
         "Rekomendacja dla biznesu: obu kanałów warto używać zamiast rezygnacji z komunikacji, "
         "a wybór między A i B można oprzeć na koszcie kontaktu. Jeśli koszty są podobne, "
         "niewielka i nieistotna przewaga kanału A przemawia raczej za nim, a dane nie dają "
         "podstaw, by preferować kanał B. Rozstrzygnięcie wymagałoby większej próby w kanale B "
         "(546 klientów z obserwowanym Y) i ustalenia, dlaczego u co czwartego klienta nie "
         "znamy wyniku.")

R.tytul("6. Kontrola krzyżowa wyników")
n_kontr, n_nie = len(rec), int((rec["zgodne"] != "tak").sum())
R.akapit("Każda część została sprawdzona przez inną osobę z grupy. Tabele części 1 policzono "
         f"od nowa z danych ({n_kontr} kontroli, niezgodności: {n_nie}). Kontrast β1 − β2 w "
         "części 3 policzono dwiema drogami: z modelu z kanałem B jako poziomem odniesienia oraz "
         "z modelu z poziomem odniesienia „brak komunikacji”, z błędem z macierzy kowariancji; "
         "oba wyniki są identyczne. Diagnostyka w części 5 liczy PPC dla dokładnie tego samego "
         "kontrastu co model docelowy i korzysta z tych samych imputacji co część 3, więc wynik "
         "przy MAR jest w całej pracy jeden. Cały kod uruchomiono od zera skryptem "
         "uruchom_wszystko.py; liczby w raporcie pochodzą z tego uruchomienia (plik "
         "wydruk_wynikow.txt).")

R.zapisz("05_raport_czesc4.docx")
