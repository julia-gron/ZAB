# ZAB – Zadanie 1B: imputacja danych (zmienna skokowa)

Czy kanał komunikacji A różnicuje szansę zakupu (Y_NEWDEPOSIT) w porównaniu z kanałem B?
Model logitowy z treści zadania, kontrast β1 − β2, przy założeniach MCAR, MAR i MNAR.

## Pliki do oddania

| Plik | Co to jest |
| --- | --- |
| `raport/raport_koncowy.pdf` | raport z rozwiązaniem (punkty 1–5 zadania) |
| `kod/zadanie1B_imputacja.py` | kod programu (Python) |
| `wyniki/wydruk_wynikow.txt` | wydruk wyników z uruchomienia kodu |

## Jak uruchomić

1. Zainstaluj pakiety: `pip install -r requirements.txt`
2. Jeśli dane leżą gdzie indziej niż `dane/kb05.csv`, zmień `DATA_PATH` na początku pliku
   `kod/zadanie1B_imputacja.py` (to jedyna ścieżka w programie).
3. Uruchom: `python kod/zadanie1B_imputacja.py` (ok. 2 minuty).

Program zapisuje tabele i wykresy w `wyniki/` oraz wydruk w `wyniki/wydruk_wynikow.txt`.
Ziarno losowe (229090) jest ustawione w sekcji 0, więc kolejne uruchomienia dają te same liczby.

## Sekcje programu

| Sekcja | Punkt zadania | Wyniki |
| --- | --- | --- |
| 0 | lokalizacja robocza, ziarno, wczytanie i rekodowanie danych | `dane_czyste.csv` |
| 1 | 1. analiza struktury, braki | `t*.csv`, `w*.png` |
| 2 | 2. zależność braku Y od cech, projekt modelu imputacyjnego | `s2_*` |
| 3 | 3. model przy MCAR (kompletne przypadki) i MAR (imputacja wielokrotna) | `s3_*` |
| 4 | 5. miara diagnostyczna (PPC), IPW, analiza wrażliwości MNAR | `s4_*` |
| 5 | kontrola krzyżowa tabel z sekcji 1 | `s4_recenzja_osoby1.csv` |
| 6 | 4. tabela zbiorcza i wykres leśny | `s5_*` |
