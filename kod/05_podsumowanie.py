# =============================================================================
# Punkt 4: podsumowanie wyników estymacji modelu docelowego w kontekście braków danych
# Uruchamiać PO 03_modele.py i 04_diagnostyka.py: łączy ich tabele wyników w jedną
# tabelę zbiorczą i rysuje wykres leśny OR A wobec B dla wszystkich metod.
# Pliki wynikowe tej części mają przedrostek s5_.
# =============================================================================
import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # zapis wykresów do plików, bez okna
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

S = importlib.import_module("00_setup")
OUT = S.WYNIKI
pd.set_option("display.width", 160)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})

# 1. TABELA ZBIORCZA: wszystkie metody w jednym formacie
s3 = pd.read_csv(OUT / "s3_tabela_wynikow.csv")
s4 = pd.read_csv(OUT / "s4_tabela_wynikow.csv")
t = pd.concat([s3, s4], ignore_index=True)
# Założenie, przy którym wynik jest poprawny
t.insert(1, "założenie", ["MCAR", "MAR"] + ["MNAR" if m.startswith("MNAR") else "MAR" for m in s4["metoda"]])
t["istotne (5%)"] = np.where((t["OR dolna 95%"] > 1) | (t["OR górna 95%"] < 1), "tak", "nie")
print("### Tabela zbiorcza: OR zakupu A wobec B")
print(t.round(4).to_string(index=False))
t.to_csv(OUT / "s5_tabela_zbiorcza.csv", index=False)

# 2. WYKRES LEŚNY OR A wobec B
KOL = {"MCAR": "#8c8c8c", "MAR": "#1f77b4", "MNAR": "#ff7f0e"}
etykiety = (t["metoda"].str.replace("Imputacja wielokrotna (MAR), ", "MI, ", regex=False)
            .str.replace(" (MAR)", "", regex=False)
            .str.replace("IPW, wagi z modelu braku z punktu 2", "IPW (wagi z modelu braku)", regex=False))
etykiety.iloc[1] = "Imputacja wielokrotna (model główny)"
poz = np.arange(len(t))[::-1]
fig, ax = plt.subplots(figsize=(9, 4.6))
for i, (_, w) in zip(poz, t.iterrows()):
    ax.errorbar(w["OR A wobec B"], i, xerr=[[w["OR A wobec B"] - w["OR dolna 95%"]],
                                             [w["OR górna 95%"] - w["OR A wobec B"]]],
                fmt="o", color=KOL[w["założenie"]], capsize=4, ms=7)
    ax.text(2.35, i, f"{w['OR A wobec B']:.2f} ({w['OR dolna 95%']:.2f}–{w['OR górna 95%']:.2f})"
            .replace(".", ","), va="center", fontsize=8)
ax.axvline(1, color="k", lw=1)
ax.set_xscale("log")
ax.set_xlim(.6, 2.3)
ax.set_xticks([.6, .8, 1, 1.25, 1.5, 2])
ax.set_xticklabels(["0,6", "0,8", "1", "1,25", "1,5", "2"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_yticks(poz)
ax.set_yticklabels(etykiety)
ax.set(xlabel="OR zakupu A wobec B (skala logarytmiczna), 95% przedział ufności",
       title="Iloraz szans zakupu A wobec B według metody")
ax.xaxis.grid(True, alpha=.3)
for zal, kol in KOL.items():
    ax.plot([], [], "o", color=kol, label=f"założenie {zal}")
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(.5, -.16))
fig.text(.01, .01, "OR > 1: szansa zakupu w kanale A większa niż w kanale B. Przedział przecinający "
         "linię OR = 1 oznacza brak istotnej różnicy.", fontsize=8, color="0.3")
plt.tight_layout(rect=(0, .03, 1, 1))
plt.savefig(OUT / "s5_w1_wykres_lesny.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nZapisano s5_tabela_zbiorcza.csv i s5_w1_wykres_lesny.png w", OUT)
