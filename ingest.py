from pathlib import Path

import chromadb
from pypdf import PdfReader
import ollama


# ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="documents"
)

DATA_FOLDER = Path("./data")


def read_pdfs():
    documents = []

    for pdf_file in DATA_FOLDER.glob("*.pdf"):
        print(f"Reading: {pdf_file.name}")

        reader = PdfReader(str(pdf_file))

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()

            if text and text.strip():
                documents.append({
                    "text": text.strip(),
                    "source": pdf_file.name,
                    "page": page_number + 1
                })

    return documents


def split_text(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_embeddings(texts):
    embeddings = []

    for text in texts:

        response = ollama.embed(
            model="nomic-embed-text",
            input=text
        )

        embeddings.append(response["embeddings"][0])

    return embeddings


def main():

    print("\nStarting document ingestion...\n")

    documents = read_pdfs()

    if not documents:
        print("No PDF files found in data folder.")
        return

    ids = []
    texts = []
    metadatas = []

    counter = 0

    for document in documents:

        chunks = split_text(document["text"])

        for chunk in chunks:

            if not chunk.strip():
                continue

            ids.append(f"chunk_{counter}")

            texts.append(chunk)

            metadatas.append({
                "source": document["source"],
                "page": document["page"]
            })

            counter += 1

    print(f"Total chunks created: {len(texts)}")

    print("Creating embeddings using Ollama...")

    embeddings = create_embeddings(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("\nDocuments successfully stored in ChromaDB!")

    print(f"Total vectors stored: {len(texts)}")


if __name__ == "__main__":
    main()