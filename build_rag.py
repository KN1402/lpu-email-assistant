from sentence_transformers import SentenceTransformer
import faiss

# Load documents

with open("data/email_guidelines.txt", "r", encoding="utf-8") as file:
    guidelines = file.read()

with open("data/email_templates.txt", "r", encoding="utf-8") as file:
    templates = file.read()

print("Documents loaded successfully!")

# Combine both documents

documents = guidelines + "\n\n" + templates


# Split document into chunks

chunks = documents.split("\n\n")

print("Chunks created:", len(chunks))

# Create embedding model

print("\nCreating embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# Convert chunks into embeddings

embeddings = model.encode(chunks)

print("Embeddings created successfully!")

# Create FAISS index

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS index created successfully!")
print("Total vectors:", index.ntotal)

# Save FAISS index

faiss.write_index(index, "email_faiss.index")

print("FAISS index saved successfully!")