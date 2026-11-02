"""Step 2 - Chunking: small overlapping word windows that keep source + page."""


def split_into_chunks(text, chunk_size=130, overlap=30):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def create_chunks(documents, chunk_size=130, overlap=30):
    all_chunks = []
    for doc in documents:
        for i, text in enumerate(split_into_chunks(doc["text"], chunk_size, overlap)):
            if len(text) < 25:
                continue
            page = doc["page"]
            all_chunks.append({
                "id": f"{doc['source']}::p{page if page is not None else 0}::c{i}",
                "text": text,
                "source": doc["source"],
                "page": page,
            })
    return all_chunks


if __name__ == "__main__":
    from .documents import load_documents
    chunks = create_chunks(load_documents())
    print("Total chunks:", len(chunks))
