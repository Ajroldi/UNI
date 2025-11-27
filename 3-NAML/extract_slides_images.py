import os
import sys
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF non installato. Installa con: pip install pymupdf", file=sys.stderr)
    sys.exit(1)

BASE_DIR = os.path.dirname(__file__)
LECTURE_DIR = os.path.join(BASE_DIR, 'note', 'Lecture October 13th')
IMG_DIR = os.path.join(BASE_DIR, 'img')

def ensure_dirs():
    os.makedirs(IMG_DIR, exist_ok=True)

def render_pdf_pages(pdf_path: str, prefix: str, dpi: int = 160, page_from: Optional[int] = None, page_to: Optional[int] = None):
    doc = fitz.open(pdf_path)
    n = doc.page_count
    start = 0 if page_from is None else max(0, page_from)
    end = n if page_to is None else min(n, page_to)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i in range(start, end):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_name = f"{prefix}_page_{i+1:02d}.png"
        out_path = os.path.join(IMG_DIR, out_name)
        pix.save(out_path)
        print(f"Salvato: {out_path}")
    doc.close()

def main():
    ensure_dirs()
    km1 = os.path.join(LECTURE_DIR, 'KM1.pdf')
    pr1 = os.path.join(LECTURE_DIR, 'PR1.pdf')
    if not os.path.isfile(km1):
        print(f"File non trovato: {km1}", file=sys.stderr)
    else:
        # Esempio: renderizza tutte le pagine; se servono solo slide specifiche, imposta page_from/page_to
        render_pdf_pages(km1, prefix='KM1', dpi=180)
    if not os.path.isfile(pr1):
        print(f"File non trovato: {pr1}", file=sys.stderr)
    else:
        render_pdf_pages(pr1, prefix='PR1', dpi=180)
    print(f"Immagini esportate in: {IMG_DIR}")

if __name__ == '__main__':
    main()