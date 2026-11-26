# =============================================================================
# Recenzja krzyżowa: Osoba 4 sprawdza tabele Osoby 1 (wyniki/t*.csv)
# Każda kontrola porównuje liczbę z tabeli z liczbą policzoną od nowa z danych.
# Wynik: wyniki/s4_recenzja_osoby1.csv (kolumna "zgodne" = tak/nie).
# =============================================================================
import importlib
import numpy as np
import pandas as pd

S = importlib.import_module("00_setup")
W = S.WYNIKI
d = S.load_clean()
N = len(d)
kontrole = []


def sprawdz(opis, z_tabeli, z_danych, tol=0.006):
    """Zapisuje jedną kontrolę; tol pozwala na zaokrąglenia w tabelach (2 miejsca)."""
    ok = bool(np.all(np.abs(np.asarray(z_tabeli, float) - np.asarray(z_danych, float)) <= tol))
    kontrole.append({"kontrola": opis, "z tabeli": z_tabeli, "z danych": z_danych,
                     "zgodne": "tak" if ok else "NIE"})


# t1: statystyki wieku
t1 = pd.read_csv(W / "t1_age_opis.csv", index_col=0)
sprawdz("t1: n ogółem = liczba klientów", t1.loc["Ogółem", "n"], N)
sprawdz("t1: suma n w kanałach = n ogółem", t1.filter(like="CONTACT", axis=0)["n"].sum(), N)
sprawdz("t1: suma n wg r = n ogółem", t1.filter(like="r=", axis=0)["n"].sum(), N)
sprawdz("t1: średni wiek ogółem", t1.loc["Ogółem", "średnia"], round(d.age.mean(), 2))
sprawdz("t1: mediana wieku w kanale B", t1.loc["CONTACT B: kanał B", "mediana"],
        d.loc[d.contact == "2", "age"].median())

# t2: cechy skokowe
t2 = pd.read_csv(W / "t2_cechy_skokowe.csv")
for zm, g in t2.groupby("zmienna", sort=False):
    sprawdz(f"t2: {zm}, suma liczności = {N}", g["n"].sum(), N)
    sprawdz(f"t2: {zm}, suma % = 100", g["%"].sum(), 100, tol=0.03)
sprawdz("t2: Y=1", t2.query("zmienna == 'y' and kategoria == '1'")["n"].iloc[0], int((d.y == 1).sum()))
sprawdz("t2: braki Y", t2.query("zmienna == 'y' and kategoria == 'brak (.)'")["n"].iloc[0], int(d.r.sum()))

# t3: tabele krzyżowe z CONTACT
for c in ["house_loan", "education", "poutcome"]:
    t3 = pd.read_csv(W / f"t3_krzyzowa_{c}.csv", header=[0, 1], index_col=0)
    n_kan = t3["n"].sum().to_numpy()
    sprawdz(f"t3 {c}: liczności kanałów (0, A, B)", list(n_kan),
            [int((d.contact == k).sum()) for k in ["0", "1", "2"]])
    sprawdz(f"t3 {c}: suma = {N}", n_kan.sum(), N)
    sprawdz(f"t3 {c}: % w kanale sumują się do 100", list(t3["% w kanale"].sum().round(1)),
            [100, 100, 100], tol=0.15)

# t4: wynik badania na obserwowanych Y
t4 = pd.read_csv(W / "t4_y_wg_contact.csv", index_col=0)
obs = d[d.r == 0]
sprawdz("t4: suma n_obs = liczba obserwowanych Y", t4["n_obs"].sum(), len(obs))
sprawdz("t4: suma Y=1 = liczba zakupów", t4["Y=1"].sum(), int(obs.y.sum()))
sprawdz("t4: odsetek Y=1 w kanale A", round(t4.iloc[1]["odsetek"], 4),
        round(obs.loc[obs.contact == "1", "y"].mean(), 4), tol=1e-4)

# t5: podgrupy (każda cecha osobno sumuje się do obserwowanych Y)
t5 = pd.read_csv(W / "t5_y_podgrupy.csv")
for cecha in ["house_loan", "poutcome"]:
    g = t5[t5.podgrupa.str.startswith(cecha)]
    sprawdz(f"t5 {cecha}: suma n_obs = obserwowane Y", g["n_obs"].sum(), len(obs))
    sprawdz(f"t5 {cecha}: suma Y=1 = zakupy", g["Y=1"].sum(), int(obs.y.sum()))

# t6-t8: braki
t6 = pd.read_csv(W / "t6_braki_zmienne.csv", index_col=0)
sprawdz("t6: braki tylko w Y", t6["braki"].drop("y").sum(), 0)
t7 = pd.read_csv(W / "t7_wzorce_brakow.csv")
sprawdz("t7: wzorce sumują się do N", t7["n"].sum(), N)
t8 = pd.read_csv(W / "t8_braki_wg_contact.csv", index_col=0)
sprawdz("t8: suma braków Y w kanałach", t8["braki Y"].sum(), int(d.r.sum()))
sprawdz("t8: suma n w kanałach", t8["n"].sum(), N)

# t9: SMD wieku A wobec B, przeliczone niezależnie
a, b = d.loc[d.contact == "1", "age"], d.loc[d.contact == "2", "age"]
t9 = pd.read_csv(W / "t9_smd.csv", header=[0, 1], index_col=0)
sprawdz("t9: SMD wieku A wobec B (pełna próba)", t9.loc["age", ("pełna próba", "A vs B")],
        round(abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2), 4), tol=1e-4)

wynik = pd.DataFrame(kontrole)
print(wynik.to_string(index=False))
print(f"\nKontroli: {len(wynik)}, niezgodnych: {(wynik['zgodne'] == 'NIE').sum()}")
wynik.to_csv(W / "s4_recenzja_osoby1.csv", index=False)
