from rag_engine import retrieve_information


query = input("Enter your question: ")

results = retrieve_information(query)


print("\n========== RAG RESULTS ==========")

for result in results:
    print("\n", result)

print("=================================")