import chromadb
import ollama


# Connect to the same ChromaDB used in ingest.py
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="documents"
)


def create_embedding(text):
    response = ollama.embed(
        model="nomic-embed-text",
        input=text
    )

    return response["embeddings"][0]


def retrieve_documents(question, number_of_results=3):

    # Convert question into vector
    question_embedding = create_embedding(question)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas


def generate_answer(question, documents, metadatas):

    context_parts = []

    for document, metadata in zip(documents, metadatas):

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

{document}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the information
provided in the context below.

If the answer is not present in the context, say:
"I couldn't find this information in the provided documents."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def ask_question(question):

    documents, metadatas = retrieve_documents(question)

    answer = generate_answer(
        question,
        documents,
        metadatas
    )

    sources = []

    for metadata in metadatas:
        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")

        sources.append(
            f"{source} - Page {page}"
        )

    return answer, sources