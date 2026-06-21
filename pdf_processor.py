import fitz  # PyMuPDF
from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file using PyMuPDF.
    Returns the full text as a single string.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += f"\n[Page {page_num + 1}]\n"
            text += page.get_text()
        doc.close()
    except Exception as e:
        raise Exception(f"Failed to read PDF: {str(e)}")
    return text


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits large text into overlapping chunks for better context retrieval.

    Args:
        text:       Full extracted text
        chunk_size: Characters per chunk
        overlap:    Characters of overlap between chunks (maintains context)

    Returns:
        List of text chunks
    """
    chunks = []
    start  = 0
    text   = text.strip()

    while start < len(text):
        end = start + chunk_size

        # Try to end at a sentence boundary
        if end < len(text):
            last_period = text.rfind('.', start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap  # overlap for continuity

    return chunks
