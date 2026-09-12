from sentence_transformers import SentenceTransformer
import faiss
import os


# -------------------------------
# 1. Load documents
# -------------------------------

files = [
    "data/email_guidelines.txt",
    "data/email_templates.txt"
]

documents = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
        documents.append(text)


# -------------------------------
# 2. Split documents into chunks
# -------------------------------

chunks = []

for document in documents:

    paragraphs = document.split("\n\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if paragraph:
            chunks.append(paragraph)


print("Documents loaded:", len(documents))
print("Chunks created:", len(chunks))


# -------------------------------
# 3. Create embeddings
# -------------------------------

print("\nCreating embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    chunks,
    normalize_embeddings=True
)


# -------------------------------
# 4. Create FAISS index
# -------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


print("FAISS index created successfully!")
print("Total vectors:", index.ntotal)


# -------------------------------
# 5. Take user query
# -------------------------------

query = input("\nEnter your question: ")


# Convert query into embedding
query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)


# -------------------------------
# 6. Retrieve relevant information
# -------------------------------

k = 3

scores, indices = index.search(query_embedding, k)


print("\n========== RETRIEVED INFORMATION ==========")

for i, score in zip(indices[0], scores[0]):

    if i != -1:
        print("\nScore:", round(float(score), 3))
        print(chunks[i])

print("============================================")