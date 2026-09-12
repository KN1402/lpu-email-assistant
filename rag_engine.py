from sentence_transformers import SentenceTransformer
import faiss


# Load saved FAISS index
index = faiss.read_index("email_faiss.index")


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load the same documents
with open("data/email_guidelines.txt", "r", encoding="utf-8") as file:
    guidelines = file.read()

with open("data/email_templates.txt", "r", encoding="utf-8") as file:
    templates = file.read()


# Combine documents
documents = guidelines + "\n\n" + templates


# Create the same chunks used while building the index
chunks = documents.split("\n\n")


def retrieve_information(query, k=3):

    # Convert user's query into embedding
    query_embedding = model.encode([query])

    # Search FAISS
    scores, indices = index.search(query_embedding, k)

    results = []

    for i in indices[0]:

        if i != -1:
            results.append(chunks[i])

    return results