# =============================================================================
# Raport końcowy: strona tytułowa, streszczenie i części 1-5 w kolejności punktów zadania
# -> raport/raport_koncowy.docx oraz raport/raport_koncowy.pdf
# Uruchamiać na końcu (po wszystkich *_raport_*.py). Wymaga pakietów python-docx
# i docxcompose. PDF powstaje przez program Word (docx2pdf albo pywin32); jeśli Worda
# nie ma, wystarczy otworzyć raport_koncowy.docx i zapisać go jako PDF.
# =============================================================================
import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Pt
from docxcompose.composer import Composer
from docx import Document
from raport_styl import Raport, f, fp, pr, RAPORT, S

W = S.WYNIKI
CZESCI = ["01_raport_czesc1.docx", "02_raport_czesc2.docx", "03_raport_czesc3.docx",
          "05_raport_czesc4.docx", "04_raport_czesc5.docx"]   # kolejność punktów 1-5 zadania
# Skład grupy - do uzupełnienia przed oddaniem
SKLAD = ["Osoba 1: [imię i nazwisko] – dane i analiza struktury (punkt 1)",
         "Osoba 2: [imię i nazwisko] – mechanizm braków danych (punkt 2)",
         "Osoba 3: [imię i nazwisko] – modele przy MCAR i MAR (punkt 3)",
         "Osoba 4: [imię i nazwisko] – diagnostyka i analiza wrażliwości (punkt 5)",
         "Osoba 5: [imię i nazwisko] – podsumowanie i raport (punkt 4)"]

t = pd.read_csv(W / "s5_tabela_zbiorcza.csv")
cc, mi = t.iloc[0], t.iloc[1]
mar = t[t["założenie"] == "MAR"]
przel = pd.read_csv(W / "s4_t6_punkt_przelamania.csv")
pA = przel[(przel["kierunek"].str.startswith("tylko kanał A")) & (przel["strona"] == "delta > 0")].iloc[0]
pBp = przel[(przel["kierunek"].str.startswith("tylko kanał B")) & (przel["strona"] == "delta > 0")].iloc[0]
glob = pd.read_csv(W / "s2_t10_testy_globalne.csv", index_col=0)
d = S.load_clean()

# --- strona tytułowa i streszczenie
R = Raport(nr_tab=1, nr_rys=1)
for tekst, rozmiar in [("Zaawansowana analityka biznesowa – metody imputacji danych", 16),
                       ("Zadanie 1, wersja B: zmienna skokowa", 14),
                       (f"Zbiór danych: {S.DATA_PATH.name}", 12)]:
    p = R.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(tekst).font.size = Pt(rozmiar)
R.doc.add_paragraph()
R.doc.add_paragraph("Skład grupy:")
for osoba in SKLAD:
    R.doc.add_paragraph(osoba)
R.doc.add_paragraph()
R.tytul("Streszczenie")
n_all = f"{len(d):,}".replace(",", " ")
n_brak = f"{int(d['r'].sum()):,}".replace(",", " ")
R.akapit("Badamy, czy kanał komunikacji A różnicuje szansę zakupu produktu w porównaniu z kanałem "
         f"B. Zbiór liczy {n_all} klientów, a wynik zakupu jest nieznany u {n_brak} z nich "
         f"({pr(d['r'].mean(), 2)}). Brak wyniku prawie nie zależy od cech klientów "
         f"(statystyka c modelu braku = {f(glob.iloc[0]['c (AUC)'], 3)}), więc analiza kompletnych "
         f"przypadków i imputacja wielokrotna dają niemal ten sam wynik. Według imputacji "
         f"wielokrotnej przy założeniu MAR szansa zakupu w kanale A jest {f(mi['OR A wobec B'])} raza "
         f"większa niż w kanale B (95% PU: {f(mi['OR dolna 95%'])}–{f(mi['OR górna 95%'])}, "
         f"p = {fp(mi['p'])}); wszystkie metody przy MCAR i MAR dają OR od "
         f"{f(min(cc['OR A wobec B'], mar['OR A wobec B'].min()))} do "
         f"{f(max(cc['OR A wobec B'], mar['OR A wobec B'].max()))} i brak istotnej różnicy. "
         f"Wniosek zmieniłby się na korzyść kanału A, gdyby klienci bez wyniku w kanale A mieli "
         f"szansę zakupu {f(pA['exp(delta)'])} raza większą niż podobni klienci z wynikiem, a "
         f"przewaga kanału B wymagałaby szansy {f(pBp['exp(delta)'])} raza większej w kanale B. "
         "Rekomendujemy wybór kanału na podstawie kosztu kontaktu; dane nie dają podstaw, by "
         "preferować kanał B.")
R.zapisz("00_strona_tytulowa.docx")

# --- złożenie części w jeden dokument (każda część od nowej strony)
glowny = Document(RAPORT / "00_strona_tytulowa.docx")
skladacz = Composer(glowny)
for plik in CZESCI:
    glowny.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    skladacz.append(Document(RAPORT / plik))
out_docx = RAPORT / "raport_koncowy.docx"
skladacz.save(out_docx)
print("Zapisano:", out_docx)

# --- PDF przez program Word, jeśli jest dostępny
out_pdf = RAPORT / "raport_koncowy.pdf"
try:
    from docx2pdf import convert
    convert(str(out_docx), str(out_pdf))
    print("Zapisano:", out_pdf)
except Exception:
    try:
        import win32com.client
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        try:
            dok = word.Documents.Open(str(out_docx.resolve()), False, True)
            dok.ExportAsFixedFormat(str(out_pdf.resolve()), 17)     # 17 = PDF
            dok.Close(False)
            print("Zapisano:", out_pdf)
        finally:
            word.Quit()
    except Exception as e:
        print("Nie udało się utworzyć PDF automatycznie (brak programu Word?):", e)
        print("Otwórz raport_koncowy.docx i zapisz go jako PDF.")
