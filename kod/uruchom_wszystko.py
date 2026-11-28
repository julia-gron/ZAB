# =============================================================================
# Uruchamia cały projekt od zera, w poprawnej kolejności, i zapisuje wydruk wyników
# (wszystko, co wypisują skrypty) do pliku wyniki/wydruk_wynikow.txt.
# Użycie (z dowolnego folderu):  python kod/uruchom_wszystko.py
# Jedyna ścieżka do zmiany przy innym położeniu danych: DATA_PATH w kod/00_setup.py.
# =============================================================================
import sys
import platform
import subprocess
import datetime
import importlib.metadata as md
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):                 # polskie znaki na każdej konsoli
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
KOD = Path(__file__).resolve().parent
WYDRUK = KOD.parent / "wyniki" / "wydruk_wynikow.txt"
SKRYPTY = [
    ("00_setup.py", "Konfiguracja: wczytanie i rekodowanie danych"),
    ("01_struktura.py", "Punkt 1: analiza struktury i braków"),
    ("02_mechanizm.py", "Punkt 2: mechanizm braków danych"),
    ("03_modele.py", "Punkt 3: model przy MCAR i MAR (imputacja wielokrotna)"),
    ("04_diagnostyka.py", "Punkt 5: diagnostyka imputacji i analiza wrażliwości"),
    ("04_recenzja_osoby1.py", "Kontrola krzyżowa tabel części 1"),
    ("05_podsumowanie.py", "Punkt 4: podsumowanie wyników"),
    ("02_raport_sekcja2.py", "Raport: część 2"),
    ("03_raport_sekcja3.py", "Raport: część 3"),
    ("05_raport_sekcja4.py", "Raport: część 4"),
    ("04_raport_sekcja5.py", "Raport: część 5"),
    ("06_raport_koncowy.py", "Raport końcowy (DOCX i PDF)"),
]
PAKIETY = ["numpy", "pandas", "scipy", "statsmodels", "patsy", "scikit-learn", "matplotlib",
           "python-docx", "docxcompose"]


def wersja(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return "brak"


WYDRUK.parent.mkdir(exist_ok=True)
with open(WYDRUK, "w", encoding="utf-8") as out:
    def pisz(tekst=""):
        print(tekst)
        out.write(tekst + "\n")

    pisz("WYDRUK WYNIKÓW – Zadanie 1B, metody imputacji danych")
    pisz(f"Data uruchomienia: {datetime.datetime.now():%Y-%m-%d %H:%M}")
    pisz(f"System: {platform.platform()} | Python {sys.version.split()[0]}")
    pisz("Pakiety: " + ", ".join(f"{p} {wersja(p)}" for p in PAKIETY))
    for plik, opis in SKRYPTY:
        pisz("\n" + "=" * 100)
        pisz(f"{plik}  –  {opis}")
        pisz("=" * 100)
        wynik = subprocess.run([sys.executable, str(KOD / plik)], cwd=KOD, capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        pisz(wynik.stdout.rstrip())
        if wynik.returncode != 0:
            pisz(wynik.stderr.rstrip())
            pisz(f"\nBŁĄD w {plik} (kod {wynik.returncode}). Przerwano.")
            sys.exit(wynik.returncode)
    pisz("\nWszystkie skrypty zakończone bez błędów.")
print("Wydruk zapisany w:", WYDRUK)
