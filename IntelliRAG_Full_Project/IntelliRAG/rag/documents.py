"""Step 1 - Document loading (PDF / TXT / MD) with page numbers."""
import io
import re
from pathlib import Path

from pypdf import PdfReader
from . import config

SUPPORTED = {".pdf", ".txt", ".md"}


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)       # re-join words split by hyphen at line end
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_bytes(name: str, data: bytes):
    """Return a list of {text, source, page} for one file."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        docs = []
        for number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                docs.append({"text": text, "source": name, "page": number})
        return docs
    if ext in (".txt", ".md"):
        text = clean_text(data.decode("utf-8", errors="ignore"))
        return [{"text": text, "source": name, "page": None}] if text else []
    raise ValueError(f"Unsupported file type: {ext or name}")


def load_file(path: Path):
    return load_bytes(path.name, path.read_bytes())


def list_files(folder: Path = None):
    folder = Path(folder or config.DATA_DIR)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)


def load_documents(folder: Path = None):
    docs = []
    for path in list_files(folder):
        try:
            docs.extend(load_file(path))
        except Exception as error:
            print(f"Could not read {path.name}: {error}")
    return docs


if __name__ == "__main__":
    loaded = load_documents()
    print("Pages/sections loaded:", len(loaded))
    for d in loaded[:3]:
        print(d["source"], d["page"], d["text"][:120].replace("\n", " "))
