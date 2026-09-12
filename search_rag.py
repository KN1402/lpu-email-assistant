from sentence_transformers import SentenceTransformer
import faiss


# Load the saved FAISS index

index = faiss.read_index("email_faiss.index")

print("FAISS index loaded successfully!")
print("Total vectors:", index.ntotal)


# Load embedding model

model = SentenceTransformer("all-MiniLM-L6-v2")


# Load the same chunks used to create the index

with open("data/email_guidelines.txt", "r", encoding="utf-8") as file:
    guidelines = file.read()

with open("data/email_templates.txt", "r", encoding="utf-8") as file:
    templates = file.read()


documents = guidelines + "\n\n" + templates

chunks = documents.split("\n\n")


# Take user's question

query = input("\nEnter your question: ")


# Convert question into embedding

query_embedding = model.encode([query])


# Search FAISS

k = 3

scores, indices = index.search(query_embedding, k)


# Display retrieved information

print("\n========== RETRIEVED INFORMATION ==========")

for i, score in zip(indices[0], scores[0]):

    if i != -1:

        print("\nScore:", round(float(score), 3))

        print(chunks[i])

print("============================================")