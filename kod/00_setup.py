# =============================================================================
# Konfiguracja - wczytanie i rekodowanie danych
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd

# 1. LOKALIZACJA ROBOCZA 
BASE = Path(__file__).resolve().parent.parent   # folder projektu (nad kod/)
DATA_PATH = BASE / "dane" / "kb05.csv"          # <- PEŁNA ścieżka do pliku z danymi
WYNIKI = BASE / "wyniki"                        # <- folder na wszystkie wyniki

assert DATA_PATH.exists(), f"Nie znaleziono pliku z danymi: {DATA_PATH}"
WYNIKI.mkdir(parents=True, exist_ok=True)       # tworzony automatycznie

# 2. ZIARNO LOSOWE
SEED = 229090
np.random.seed(SEED)

# 3. ETYKIETY I POZIOMY REFERENCYJNE
# Pierwszy poziom na liście = poziom referencyjny w modelu.
CONTACT_LAB = {0: "0: brak komunikacji", 1: "A: kanał A", 2: "B: kanał B"}
CONTACT_LEVELS = ["2", "0", "1"]                              # ref = 2 (kanał B)
HL_LEVELS = ["yes", "no"]                                     # ref = yes
POUTCOME_LEVELS = ["unknown", "failure", "success", "other"]  # ref = unknown
EDU_LEVELS = ["secondary", "primary", "tertiary", "unknown"]  # ref = secondary
LEVELS = {"contact": CONTACT_LEVELS, "house_loan": HL_LEVELS,
          "poutcome": POUTCOME_LEVELS, "education": EDU_LEVELS}
KOLUMNY = ["y", "contact", "house_loan", "age", "education", "poutcome"]


# 4. WCZYTANIE DANYCH
def load_raw() -> pd.DataFrame:
    """Surowe dane jako tekst: nic nie jest po cichu zamieniane na brak."""
    # sep=None: separator (przecinek lub średnik) wykrywany automatycznie
    return pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False,
                       sep=None, engine="python")


def load_clean() -> pd.DataFrame:
    """Dane po rekodowaniu: kategorie z poziomami ref, y (NaN = brak), r."""
    d = load_raw()
    # Nazwy kolumn małymi literami (eksport z SAS bywa wielkimi): y_newdeposit -> y
    d.columns = d.columns.str.strip().str.lower()
    d = d.rename(columns={"y_newdeposit": "y"})
    brak_kolumn = set(KOLUMNY) - set(d.columns)
    assert not brak_kolumn, f"W pliku brakuje kolumn: {sorted(brak_kolumn)}"

    # Brak danych = puste pole lub kropka (zapis SAS), w KAŻDEJ kolumnie.
    # "unknown" NIE jest brakiem, to zwykła kategoria.
    d = d[KOLUMNY].apply(lambda s: s.str.strip()).replace({"": np.nan, ".": np.nan})

    # Zmienne liczbowe (tekst, którego nie da się zamienić na liczbę, zatrzyma program)
    d["y"] = pd.to_numeric(d["y"])
    d["age"] = pd.to_numeric(d["age"])
    assert d["y"].dropna().isin([0, 1]).all(), "Y musi mieć wartości 0 lub 1"

    # Cechy skokowe jako kategorie z ustalonymi poziomami (contact NIE jako liczba).
    # Puste pole zostaje brakiem, a nieznana etykieta (np. literówka) zatrzymuje program.
    for kol, poziomy in LEVELS.items():
        s = d[kol].str.lower() if kol != "contact" else d[kol]
        nieznane = set(s.dropna()) - set(poziomy)
        assert not nieznane, f"Nieznane kategorie w {kol}: {sorted(nieznane)}"
        d[kol] = pd.Categorical(s, categories=poziomy)
    assert d["contact"].notna().all(), "contact (grupa losowania) nie może mieć braków"

    d["r"] = d["y"].isna().astype(int)           # r = 1 oznacza brak Y
    return d[KOLUMNY + ["r"]]


# 5. URUCHOMIENIE BEZPOŚREDNIE: zapis danych czystych
if __name__ == "__main__":
    df = load_clean()
    df.to_csv(WYNIKI / "dane_czyste.csv", index=False)
    print("OK:", df.shape, "| braki Y:", int(df["r"].sum()))

