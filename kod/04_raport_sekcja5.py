# =============================================================================
# Część 5 raportu (diagnostyka imputacji, analiza wrażliwości) -> raport/04_raport_czesc5.docx
# Uruchamiać PO 04_diagnostyka.py. Liczby są czytane z plików s4_*.csv, więc raport
# zgadza się z wydrukiem. Wygląd jak w części 1 (moduł raport_styl.py).
# Numeracja: część 4 kończy się na tabeli 16 i rysunku 9.
# =============================================================================
import numpy as np
import pandas as pd
from raport_styl import Raport, f, fp, S

W = S.WYNIKI
R = Raport(nr_tab=17, nr_rys=10)

efekt = pd.read_csv(W / "s4_t1_miary_efektywnosci.csv", index_col=0)["wartość"]
ppc = pd.read_csv(W / "s4_t2_ppc.csv")
war = pd.read_csv(W / "s4_t3_warianty_imputacji.csv")
wagi = pd.read_csv(W / "s4_t4_ipw_wagi.csv").iloc[0]
przel = pd.read_csv(W / "s4_t6_punkt_przelamania.csv")
siatka = pd.read_csv(W / "s4_t7_mnar_siatka.csv")
s4 = pd.read_csv(W / "s4_tabela_wynikow.csv").set_index("metoda")
s3 = pd.read_csv(W / "s3_tabela_wynikow.csv")
odsetki = pd.read_csv(W / "s3_odsetek_y_imputowane.csv", index_col=0)
M = int(efekt["liczba imputacji m"])
ppc_gl = ppc[ppc.iloc[:, 0].str.startswith("model z punktu 2")]
ppc_zly = ppc[ppc.iloc[:, 0].str.startswith("model celowo")].iloc[0]
ipw = s4.loc["IPW (MAR), wagi z modelu braku z punktu 2"]

R.tytul("Część 5: diagnostyka imputacji i analiza wrażliwości")
R.akapit("W tej części odpowiadamy na punkt 5 zadania: proponujemy miarę diagnostyczną do oceny "
         "imputacji w kontekście analizowanego modelu i ją uzasadniamy. Następnie sprawdzamy, "
         "jak mocny jest wniosek o braku różnicy między kanałami A i B. Diagnostyka dotyczy tych "
         f"samych {M} imputacji, na których oparto wynik w części 3.")

# --- 1. Miary efektywności
R.tytul("1. Miary efektywności imputacji wielokrotnej")
R.akapit("Pierwszą grupą miar są wielkości wynikające z reguł Rubina: względny wzrost wariancji "
         "(o ile braki zwiększyły wariancję oceny), frakcja brakującej informacji FMI (jaką część "
         "informacji o parametrze odebrały braki) oraz efektywność oszacowania przy skończonej "
         "liczbie imputacji, (1 + FMI/m)⁻¹ (tabela 17).")
t = pd.DataFrame({"Miara": ["Liczba imputacji m", "Względny wzrost wariancji",
                            "Frakcja brakującej informacji (FMI)", "Efektywność (1 + FMI/m)⁻¹"],
                  "Wartość": [str(M), f(efekt["względny wzrost wariancji"], 3), f(efekt.iloc[2], 3),
                              f(efekt.iloc[3], 4)]})
R.tabela(t, "Miary efektywności imputacji wielokrotnej dla β1 − β2.")
R.akapit(f"Braki zwiększyły wariancję oceny β1 − β2 o {round(100 * efekt['względny wzrost wariancji'])}% "
         f"i odebrały około {round(100 * efekt.iloc[2])}% informacji o różnicy między kanałami. "
         f"Przy {M} imputacjach efektywność wynosi {f(efekt.iloc[3], 4)}, więc dalsze zwiększanie "
         "liczby imputacji nie zmieniłoby wyniku. Miary te mówią jednak tylko, ile informacji "
         "utracono, a nie czy model imputacyjny jest dobry. Do tego służy miara z punktu 2.")

# --- 2. PPC
R.tytul("2. Proponowana miara: predykcyjna wartość p a posteriori dla β1 − β2")
R.akapit("Proponujemy predykcyjną wartość p a posteriori (ang. posterior predictive check, PPC; "
         "Nguyen, Lee i Carlin, 2015) liczoną dla tej samej wielkości, o którą pyta zadanie, "
         "czyli dla różnicy β1 − β2 w modelu docelowym. Procedura ma trzy kroki. (1) W każdej "
         "imputacji, z tymi samymi wylosowanymi parametrami modelu imputacyjnego, losujemy Y "
         "także dla klientów z obserwowanym Y; tak powstają dane replikowane. (2) W każdej "
         "imputacji szacujemy model docelowy dwa razy: na danych uzupełnionych (obserwowane Y + "
         "imputowane) i na danych replikowanych. (3) PPC p to odsetek imputacji, w których "
         "β1 − β2 z danych replikowanych jest większe niż z danych uzupełnionych. Jeśli model "
         "imputacyjny dobrze odtwarza zależność zakupu od kanału, oba wyniki różnią się tylko "
         "losowo i PPC p jest bliskie 0,5. Wartości bliskie 0 lub 1 oznaczają, że model "
         "imputacyjny systematycznie zawyża lub zaniża kontrast.")
R.akapit("Uzasadnienie wyboru. Po pierwsze, miara ocenia model imputacyjny dokładnie pod kątem "
         "wniosku, który z niego wyciągamy, a nie ogólnej jakości dopasowania. Po drugie, Y jest "
         "zmienną binarną, więc porównanie rozkładów wartości imputowanych i obserwowanych "
         "niewiele mówi: imputowane są same zera i jedynki. Po trzecie, przy MAR odsetek zakupów "
         "wśród imputowanych może legalnie różnić się od obserwowanego (w kanale B: "
         f"{f(100 * odsetki.loc[2, 'imputowane Y'], 1)}% wobec {f(100 * odsetki.loc[2, 'obserwowane Y'], 1)}%, "
         "część 3), więc samo porównanie odsetków nie odróżnia dobrej imputacji od złej. PPC "
         "porównuje dane z modelem przy tych samych cechach klientów.")
tp = pd.DataFrame({"Model imputacyjny": ["z punktu 2"] * len(ppc_gl) + ["celowo błędny (bez CONTACT)"],
                   "Statystyka": list(ppc_gl["statystyka"].str.replace("beta1 - beta2", "β1 − β2")
                                      .str.replace("Y=1", "Y = 1")) + ["β1 − β2 (A wobec B)"],
                   "Średnia: dane uzupełnione": [f(x, 3) for x in ppc_gl["średnia uzupełnione"]] + [f(ppc_zly["średnia uzupełnione"], 3)],
                   "Średnia: dane replikowane": [f(x, 3) for x in ppc_gl["średnia replikowane"]] + [f(ppc_zly["średnia replikowane"], 3)],
                   "PPC p": [f(x, 3) for x in ppc_gl["PPC p"]] + [f(ppc_zly["PPC p"], 3)]})
nr = R.tabela(tp, "Predykcyjne wartości p a posteriori. Wartość bliska 0,5 oznacza dobre "
                  "dopasowanie, bliska 0 lub 1 – złe.", num_od=2)
nr_r = R.rysunek("s4_w1_ppc.png", "β1 − β2 na danych uzupełnionych i replikowanych w każdej imputacji.")
R.akapit(f"Dla modelu imputacyjnego z części 2 PPC p dla kontrastu wynosi {f(ppc_gl.iloc[0]['PPC p'])}, "
         f"a dla odsetków zakupów w kanałach od {f(ppc_gl['PPC p'].iloc[1:].min())} do "
         f"{f(ppc_gl['PPC p'].iloc[1:].max())} (tabela {nr}). Punkty na rysunku {nr_r} leżą po obu "
         "stronach przekątnej. Model imputacyjny odtwarza więc zarówno kontrast A wobec B, jak i "
         "poziom zakupów w każdym kanale.")
R.akapit("Żeby sprawdzić, czy miara potrafi wykryć zły model, powtórzyliśmy procedurę dla modelu "
         "celowo błędnego, bez zmiennej CONTACT. Średni kontrast na danych replikowanych "
         f"({f(ppc_zly['średnia replikowane'], 3)}) jest wtedy wielokrotnie mniejszy niż na "
         f"uzupełnionych ({f(ppc_zly['średnia uzupełnione'], 3)}), a PPC p spada do "
         f"{f(ppc_zly['PPC p'])}. Spadek nie jest skrajny, bo ocena β1 − β2 ma duży błąd losowy: "
         "w kanale B jest tylko 546 obserwowanych Y. Ograniczenia miary: PPC p nie ma klasycznych "
         "progów typu 0,05 i nie jest wartością p testu, a jego czułość spada, gdy braków jest "
         "dużo albo statystyka jest mało precyzyjna. Dlatego raportujemy je razem z PPC dla "
         "odsetków zakupów w kanałach i z miarami z punktu 1.")

# --- 3. Warianty i IPW
R.tytul("3. Czy wynik zależy od modelu imputacyjnego i od metody korekty?")
tw = pd.DataFrame({"Wariant modelu imputacyjnego": war["wariant"],
                   "β1 − β2": [f(x, 3) for x in war["beta1 - beta2"]],
                   "Błąd": [f(x, 3) for x in war["błąd"]],
                   "OR A wobec B": [f(x) for x in war["OR A wobec B"]],
                   "p": [fp(x) for x in war["p"]],
                   "FMI": [f(x) for x in war["FMI"]],
                   "PPC p": [f(x) for x in war["PPC p"]]})
nr = R.tabela(tw, f"Porównanie modeli imputacyjnych (każdy wariant: {M} imputacji).")
R.akapit("Usunięcie interakcji CONTACT × HOUSING_LOAN oraz dodanie interakcji CONTACT × AGE, która "
         "w części 2 była istotna w modelu braku, zmieniają ocenę β1 − β2 o mniej niż "
         f"{f(np.ceil(100 * (war['beta1 - beta2'] - war['beta1 - beta2'].iloc[0]).abs().max()) / 100)}, "
         "czyli o niewielki ułamek jej błędu standardowego. Wszystkie warianty mają PPC p bliskie "
         f"0,5 (tabela {nr}).")
R.akapit("Drugą metodą korekty przy MAR jest ważenie odwrotnością prawdopodobieństwa obserwacji Y "
         "(IPW): klienci z obserwowanym Y dostają wagę 1 / P(Y obserwowane | X), stabilizowaną "
         "odsetkiem obserwowanych Y. Prawdopodobieństwa pochodzą z modelu braku z części 2. Wagi "
         f"są bliskie 1 (od {f(wagi['min'], 3)} do {f(wagi['max'], 3)}), bo model braku prawie nie "
         "odróżnia klientów z brakiem Y od pozostałych. Model docelowy z wagami i odpornymi "
         f"błędami standardowymi daje OR = {f(ipw['OR A wobec B'])} "
         f"({f(ipw['OR dolna 95%'])}–{f(ipw['OR górna 95%'])}), czyli prawie tyle samo co "
         f"imputacja ({f(s3.iloc[1]['OR A wobec B'])}). Wniosek nie zależy więc od sposobu korekty "
         "przy MAR.")

# --- 4. MNAR
R.tytul("4. Analiza wrażliwości na braki nielosowe (MNAR)")
R.akapit("Dane nie pozwalają odróżnić MAR od MNAR: nie wiemy, czy klienci bez Y kupowali częściej "
         "czy rzadziej niż podobni klienci z Y. Dlatego sprawdzamy, jak bardzo musiałoby się "
         "różnić ich zachowanie, żeby zmienił się wniosek. W imputacji dodajemy do logitu szansy "
         "zakupu klientów bez Y przesunięcie δA w kanale A i δB w kanale B (w grupie bez "
         "komunikacji δ = 0). exp(δ) mówi, ile razy szansa zakupu klienta bez Y jest większa od "
         "szansy podobnego klienta z Y; δ = 0 oznacza MAR. Dla każdej wartości δ wykonujemy pełną "
         "imputację i łączymy wyniki regułami Rubina. Punkt przełamania to najmniejsze "
         "odchylenie od MAR, przy którym 95% przedział ufności OR A wobec B przestaje zawierać 1.")
nr_l = R.rysunek("s4_w2_mnar_linie.png", "OR A wobec B przy przesunięciu tylko w kanale A i tylko "
                 "w kanale B; pomarańczowe linie: punkty przełamania.")
tq = pd.DataFrame({"Odchylenie od MAR": [f"{k}, {s.replace('delta', 'δ')}" for k, s in zip(przel["kierunek"], przel["strona"])],
                   "δ": ["–" if pd.isna(x) else f(x) for x in przel["delta"]],
                   "exp(δ)": ["–" if pd.isna(x) else f(x) for x in przel["exp(delta)"]],
                   "OR A wobec B": ["–" if pd.isna(x) else f(x) for x in przel["OR A wobec B"]],
                   "Wniosek po przełamaniu": przel["wniosek"].str.replace("delta", "δ")})
nr = R.tabela(tq, "Punkty przełamania: najmniejsze odchylenie od MAR, przy którym 95% przedział "
                  "ufności OR A wobec B przestaje zawierać 1.", num_od=1, num_do=4)
nr_m = R.rysunek("s4_w3_mnar_mapa.png", "OR zakupu A wobec B przy jednoczesnym przesunięciu w "
                 "kanale A i B.")
pA = przel[(przel["kierunek"].str.startswith("tylko kanał A")) & (przel["strona"] == "delta > 0")].iloc[0]
pBm = przel[(przel["kierunek"].str.startswith("tylko kanał B")) & (przel["strona"] == "delta < 0")].iloc[0]
pBp = przel[(przel["kierunek"].str.startswith("tylko kanał B")) & (przel["strona"] == "delta > 0")].iloc[0]
R.akapit(f"Wniosek z części 3 („brak istotnej różnicy między kanałami”) jest wrażliwy na MNAR w "
         f"jednym kierunku (rysunek {nr_l}, tabela {nr}). Wystarczy, że klienci bez Y w kanale A "
         f"mają szansę zakupu {f(pA['exp(delta)'])} raza większą niż podobni klienci z Y "
         f"(δA = {f(pA['delta'])}), albo że klienci bez Y w kanale B mają ją "
         f"{f(1 / pBm['exp(delta)'])} raza mniejszą (δB = {f(pBm['delta'])}), a kanał A staje się "
         "istotnie lepszy. Odwrotny wniosek, że kanał B jest istotnie lepszy, wymaga dużo "
         f"silniejszego MNAR: δB = {f(pBp['delta'])}, czyli szansy {f(pBp['exp(delta)'])} raza "
         "większej. Mapa na rysunku "
         f"{nr_m} pokazuje to samo dla jednoczesnych przesunięć: jednakowe przesunięcie w obu "
         "kanałach (przekątna) prawie nie zmienia OR, bo działa na A i B tak samo.")
R.akapit("Czy takie odchylenia są wiarygodne? Status klienta jest nieznany w t1, więc brak Y może "
         "wynikać np. z braku kontaktu po kampanii albo z niedokończonego procesu zakupu. Różnica "
         "szans rzędu 1,5–2 razy między klientami bez Y i z Y nie jest nieprawdopodobna, natomiast "
         "różnica rzędu 4–5 razy, potrzebna do odwrócenia wniosku na korzyść kanału B, jest mało "
         "realistyczna. Dane nie dają więc podstaw, by uznać kanał B za lepszy, a ocena „A i B nie "
         "różnią się” jest wnioskiem ostrożnym, który przy umiarkowanym MNAR przesunąłby się na "
         "korzyść kanału A.")

# --- 5. Wnioski
R.tytul("5. Wnioski")
R.akapit(f"Proponowaną miarą diagnostyczną jest PPC p dla β1 − β2. Dla przyjętego modelu "
         f"imputacyjnego wynosi {f(ppc_gl.iloc[0]['PPC p'])}, a dla odsetków zakupów w kanałach "
         "jest bliskie 0,5, więc model imputacyjny jest dobrze dopasowany do wielkości, o którą "
         "pyta zadanie. Wynik przy MAR jest stabilny: różne modele imputacyjne i IPW dają OR A "
         "wobec B około 1,2, zawsze z przedziałem ufności zawierającym 1. Wniosek o braku różnicy "
         "jest wrażliwy na umiarkowane MNAR na korzyść kanału A, natomiast przewaga kanału B "
         "wymagałaby nierealistycznie silnego MNAR.")

R.tytul("Literatura")
R.akapit("Nguyen, C. D., Lee, K. J., Carlin, J. B. (2015). Posterior predictive checking of multiple "
         "imputation models. Biometrical Journal, 57(4), 676–694.")
R.akapit("Little, R. J. A., Rubin, D. B. (2002). Statistical Analysis with Missing Data (2nd ed.). "
         "Hoboken: John Wiley & Sons.")

R.zapisz("04_raport_czesc5.docx")
