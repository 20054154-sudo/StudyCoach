"""Unicode PDF helpers shared by report and notes generators."""

from pathlib import Path
import unicodedata

from fpdf import FPDF

FONT_DIR = Path(__file__).parent.parent / "assets" / "fonts"
FONT_FILES = {
    "": FONT_DIR / "Arial.ttf",
    "B": FONT_DIR / "Arial-Bold.ttf",
    "I": FONT_DIR / "Arial-Italic.ttf",
    "BI": FONT_DIR / "Arial-BoldItalic.ttf",
}

# Arial does not cover every mathematical symbol. These common symbols get a
# readable fallback instead of being silently discarded.
SYMBOL_FALLBACKS = {
    "σ": "sigma", "Σ": "Sigma", "μ": "mu", "Μ": "Mu",
    "∑": "somma", "∫": "integrale", "√": "radice",
    "≤": "<=", "≥": ">=", "→": "->", "←": "<-",
    "✓": "OK", "✅": "OK", "⚠": "!", "€": "Euro",
    "•": "-", "–": "-", "—": "-", "…": "...",
    "’": "'", "‘": "'", "“": '"', "”": '"',
}


def pdf_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value))
    for source, target in SYMBOL_FALLBACKS.items():
        text = text.replace(source, target)
    return text


def register_unicode_fonts(pdf: FPDF) -> None:
    for style, path in FONT_FILES.items():
        if not path.exists():
            raise RuntimeError(f"Font Unicode mancante: {path}")
        pdf.add_font("Arial", style, str(path))
