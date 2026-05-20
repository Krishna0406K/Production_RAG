import os
from pathlib import Path
from typing import List, Dict

def _extract_pdf(path: str) -> str:
    try:
        import fitz
    except ImportError:
        raise ImportError('PyMuPDF is required for PDF ingestion.\nInstall it with:  pip install pymupdf')
    text_parts = []
    doc = None
    try:
        doc = fitz.open(path)
        page_count = len(doc)
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text('text')
            if page_text.strip():
                text_parts.append(f'[Page {page_num}]\n{page_text}')
        full_text = '\n\n'.join(text_parts)
        print(f"[DocumentLoader] PDF '{Path(path).name}': {page_count} pages, {len(full_text)} chars extracted.")
        return full_text
    except Exception as e:
        print(f"[DocumentLoader] Error reading PDF path '{path}': {e}")
        raise e
    finally:
        if doc is not None:
            doc.close()

def _extract_docx(path: str) -> str:
    try:
        import docx
    except ImportError:
        raise ImportError('python-docx is required for .docx ingestion.\nInstall it with:  pip install python-docx')
    doc = docx.Document(path)
    parts = []
    for element in doc.element.body:
        tag = element.tag.split('}')[-1]
        if tag == 'p':
            para_text = element.text_content().strip() if hasattr(element, 'text_content') else ''
            para = docx.text.paragraph.Paragraph(element, doc)
            para_text = para.text.strip()
            if para_text:
                parts.append(para_text)
        elif tag == 'tbl':
            table = docx.table.Table(element, doc)
            for row in table.rows:
                row_text = '  '.join((cell.text.strip() for cell in row.cells if cell.text.strip()))
                if row_text:
                    parts.append(row_text)
    full_text = '\n\n'.join(parts)
    print(f"[DocumentLoader] DOCX '{Path(path).name}': {len(full_text)} chars extracted.")
    return full_text

def _extract_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    print(f"[DocumentLoader] TXT/MD '{Path(path).name}': {len(text)} chars.")
    return text
_EXTRACTORS = {'.pdf': _extract_pdf, '.docx': _extract_docx, '.doc': _extract_docx, '.txt': _extract_txt, '.md': _extract_txt, '.rst': _extract_txt}

def load_document(source: str) -> Dict[str, str]:
    p = Path(source)
    is_file = p.suffix.lower() in _EXTRACTORS or (len(source) < 300 and p.exists())
    if is_file:
        if not p.exists():
            raise FileNotFoundError(f'File not found: {source}')
        ext = p.suffix.lower()
        if ext not in _EXTRACTORS:
            supported = ', '.join(_EXTRACTORS.keys())
            raise ValueError(f"Unsupported file type '{ext}'. Supported: {supported}")
        text = _EXTRACTORS[ext](str(p))
        source_name = p.name
    else:
        text = source
        source_name = 'inline_text'
    if not text.strip():
        raise ValueError(f"No text could be extracted from '{source}'.")
    return {'text': text, 'source': source_name}

def load_documents(sources: List[str]) -> List[Dict[str, str]]:
    results = []
    for src in sources:
        try:
            results.append(load_document(src))
        except Exception as e:
            print(f"[DocumentLoader] ⚠ Skipping '{src}': {e}")
    return results
