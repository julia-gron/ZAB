# ZAB – Zadanie 1B: imputacja danych (zmienna skokowa)

Czy kanał komunikacji A różnicuje szansę zakupu (Y_NEWDEPOSIT) w porównaniu z kanałem B?
Model logitowy z treści zadania, kontrast β1 − β2, przy założeniach MCAR, MAR i MNAR.

## Pliki do oddania

| Plik | Co to jest |
| --- | --- |
| `raport/raport_koncowy.pdf` | raport z rozwiązaniem (punkty 1–5 zadania) |
| `kod/` | kod programu (Python) |
| `wyniki/wydruk_wynikow.txt` | wydruk wyników z uruchomienia kodu |

## Jak uruchomić

1. Zainstaluj pakiety: `pip install -r requirements.txt`
2. Jeśli dane leżą gdzie indziej niż `dane/kb05.csv`, zmień `DATA_PATH` w `kod/00_setup.py`
   (to jedyna ścieżka w projekcie).
3. Uruchom wszystko: `python kod/uruchom_wszystko.py` (ok. 2 minuty).

Skrypt wykonuje pliki w poprawnej kolejności, zapisuje tabele i wykresy w `wyniki/` oraz
wydruk w `wyniki/wydruk_wynikow.txt`. Ziarno losowe (229090) jest ustawione w `00_setup.py`,
więc kolejne uruchomienia dają te same liczby.

## Kolejność i zawartość skryptów

| Skrypt | Punkt zadania | Wyniki |
| --- | --- | --- |
| `00_setup.py` | ścieżki, ziarno, wczytanie i rekodowanie danych | `dane_czyste.csv` |
| `01_struktura.py` | 1. analiza struktury, braki | `t*.csv`, `w*.png` |
| `02_mechanizm.py` | 2. zależność braku Y od cech, projekt modelu imputacyjnego | `s2_*` |
| `03_modele.py` | 3. model przy MCAR (kompletne przypadki) i MAR (imputacja wielokrotna) | `s3_*` |
| `04_diagnostyka.py` | 5. miara diagnostyczna (PPC), IPW, analiza wrażliwości MNAR | `s4_*` |
| `04_recenzja_osoby1.py` | kontrola krzyżowa tabel części 1 | `s4_recenzja_osoby1.csv` |
| `05_podsumowanie.py` | 4. tabela zbiorcza i wykres leśny | `s5_*` |

Raport (`raport/`) zawiera tabele i wykresy z folderu `wyniki/`.
