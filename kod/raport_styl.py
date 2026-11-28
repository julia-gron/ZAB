# =============================================================================
# Wspólny wygląd części raportu (używany przez skrypty *_raport_*.py).
# Szablonem jest raport części 1 (raport/01_raport_czesc1.docx): te same marginesy,
# czcionki i odstępy; nagłówki zwykłym tekstem, podpisy nad tabelami i rysunkami,
# tabele z siatką (Aptos Narrow 11), liczby wyrównane do prawej.
# Numeracja tabel i rysunków jest ciągła w całym raporcie, dlatego każda część
# podaje numer swojej pierwszej tabeli i pierwszego rysunku.
# =============================================================================
import importlib
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

S = importlib.import_module("00_setup")
W = S.WYNIKI
RAPORT = S.BASE / "raport"
SZABLON = RAPORT / "01_raport_czesc1.docx"


def f(x, n=2):
    """Liczba w zapisie polskim, np. 0,18."""
    return f"{x:.{n}f}".replace(".", ",")


def ft(x, n=1):
    """Liczba z separatorem tysięcy (spacja), np. 12 815,6."""
    return f"{x:,.{n}f}".replace(",", " ").replace(".", ",")


def fp(p):
    """Wartość p: '< 0,001' albo trzy miejsca po przecinku."""
    return "< 0,001" if p < .001 else f(p, 3)


def pr(x, n=1):
    """Odsetek (0-1) jako procent, np. 13,6%."""
    return f(100 * x, n) + "%"


class Raport:
    """Dokument jednej części raportu z ciągłą numeracją tabel i rysunków."""

    def __init__(self, nr_tab, nr_rys):
        self.doc = Document(SZABLON)
        body = self.doc.element.body
        for el in list(body):                      # zostawiamy tylko ustawienia strony
            if not el.tag.endswith("sectPr"):
                body.remove(el)
        self.nr_tab, self.nr_rys = nr_tab, nr_rys

    def tytul(self, tekst):
        """Nagłówek w stylu części 1: zwykły tekst w osobnym akapicie."""
        self.doc.add_paragraph(tekst)

    def akapit(self, tekst):
        p = self.doc.add_paragraph(tekst)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(12)
        return p

    def tabela(self, df, podpis, num_od=1, num_do=None):
        """Podpis nad tabelą; kolumny num_od..num_do-1 wyrównane do prawej (liczby).
        Zwraca numer tabeli."""
        nr = self.nr_tab
        self.doc.add_paragraph(f"Tabela {nr}. {podpis}")
        self.nr_tab += 1
        t = self.doc.add_table(rows=1, cols=len(df.columns))
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
                    elif num_od <= c_i < (num_do or len(df.columns)):
                        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                    for r in p.runs:
                        r.font.name = "Aptos Narrow"
                        r.font.size = Pt(11)
                        r.bold = r_i == 0
        self.doc.add_paragraph()
        return nr

    def rysunek(self, plik, podpis, szer=16):
        """Podpis nad rysunkiem, jak w części 1. Zwraca numer rysunku."""
        nr = self.nr_rys
        self.doc.add_paragraph(f"Rysunek {nr}. {podpis}")
        self.nr_rys += 1
        self.doc.add_picture(str(W / plik), width=Cm(szer))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        return nr

    def zapisz(self, nazwa):
        out = RAPORT / nazwa
        self.doc.save(out)
        print("Zapisano:", out, f"| następna tabela: {self.nr_tab}, następny rysunek: {self.nr_rys}")
