import os
from google import genai


# Get API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")


# Create Gemini client
client = genai.Client(api_key=api_key)


# Send a simple request
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain in one sentence why professional emails are important."
)


# Display Gemini response
print("\n========== GEMINI RESPONSE ==========")
print(response.text)
print("=====================================")