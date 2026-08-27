from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst"}


class DocumentationRAG:

    def __init__(self):
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name="documentation"
        )

    def load_documents(self, file_paths: list[str]):
        documents = []

        for file_path in file_paths:
            path = Path(file_path)

            if not path.is_file():
                print(f"Skipping invalid file: {file_path}")
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                print(f"Unsupported file type: {path.name}")
                continue

            with open(
                path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:
                text = f.read()

            documents.append({
                "text": text,
                "file": path.name,
                "path": str(path)
            })

        return documents

    def chunk_text(self, text: str, chunk_size=500, overlap=50):
        words = text.split()
        chunks = []

        start = 0

        while start < len(words):
            end = start + chunk_size

            chunk = " ".join(words[start:end])

            if chunk.strip():
                chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

    def build_index(self, file_paths: list[str]):
        documents = self.load_documents(file_paths)

        ids = []
        chunks = []
        metadatas = []

        chunk_id = 0

        for document in documents:

            document_chunks = self.chunk_text(
                document["text"]
            )

            for chunk in document_chunks:

                ids.append(str(chunk_id))
                chunks.append(chunk)

                metadatas.append({
                    "file": document["file"],
                    "path": document["path"],
                    "chunk_id": chunk_id
                })

                chunk_id += 1

        if not chunks:
            print("No documents found.")
            return

        embeddings = self.embedding_model.encode(
            chunks
        ).tolist()

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Indexed {len(chunks)} chunks.")

    def retrieve(self, query: str, top_k=5):

        query_embedding = self.embedding_model.encode(
            query
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved = []

        for i, document in enumerate(
            results["documents"][0]
        ):
            retrieved.append({
                "content": document,
                "metadata": results["metadatas"][0][i]
            })

        return retrieved


# -------------------------
# Runtime
# -------------------------

rag = DocumentationRAG()

user_input = input(
    "Enter document paths separated by commas:\n"
)

file_paths = [
    path.strip()
    for path in user_input.split(",")
]

rag.build_index(file_paths)

while True:

    query = input("\nAsk a question (or 'exit'): ")

    if query.lower() == "exit":
        break

    results = rag.retrieve(query)

    print("\n--- Retrieved Information ---")

    for result in results:

        print(
            f"\nSource: {result['metadata']['file']}"
        )

        print(result["content"])