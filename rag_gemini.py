import os

from google import genai
from rag_engine import retrieve_information


# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=api_key)


# Take user's question
query = input("What do you need help with? ")


# Retrieve information using RAG
results = retrieve_information(query)


# Combine retrieved information
context = "\n\n".join(results)


# Create prompt for Gemini
prompt = f"""
You are a university email assistant.

Use the following information retrieved from the university
knowledge base to answer the user's request.

KNOWLEDGE BASE:
{context}

USER REQUEST:
{query}

Give a clear and professional response.
Do not invent university rules that are not present in the knowledge base.
"""


# Send request to Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)


# Display result
print("\n========== GEMINI + RAG RESPONSE ==========")
print(response.text)
print("============================================")