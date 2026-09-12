# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# # Load dataset
# data = pd.read_csv("data/intents.csv")

# # Input and output
# X = data["text"]
# y = data["intent"]

# # Convert text into numbers
# vectorizer = TfidfVectorizer()
# X_vectorized = vectorizer.fit_transform(X)

# # Train model
# model = LogisticRegression()
# model.fit(X_vectorized, y)

# # Take input from user
# user_input = input("Enter your message: ")

# # Convert user message into numbers
# user_vector = vectorizer.transform([user_input])

# # Predict intent
# prediction = model.predict(user_vector)

# print("Predicted intent:", prediction[0])

# import joblib

# # Load saved model and vectorizer
# model = joblib.load("email_intent_model.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")


# # Take user input
# message = input("Enter your message: ")


# # Convert message into numbers
# message_vectorized = vectorizer.transform([message])


# # Predict intent
# prediction = model.predict(message_vectorized)


# print("Predicted intent:", prediction[0])

import joblib
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rag_engine import retrieve_information
from gmail_sender import send_email


# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv(".env", override=True)


# ==========================================================
# GEMINI CLIENT
# ==========================================================

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=60000
    )
)


# ==========================================================
# FUNCTION: PREVIEW AND SEND EMAIL
# ==========================================================

def preview_and_send(email, default_recipient=None):

    print("\n========== EMAIL PREVIEW ==========")
    print(email)
    print("===================================")

    choice = input("\nDo you want to send this email? (yes/no): ").lower()

    if choice == "yes":

        if default_recipient:
            recipient = default_recipient
        else:
            recipient = input("Enter recipient email address: ")

        # Extract subject
        lines = email.splitlines()

        subject = "LPU Academic Email"

        for line in lines:
            if line.lower().startswith("subject:"):
                subject = line.split(":", 1)[1].strip()
                break

        # Remove Subject line from email body
        body = "\n".join(
            line for line in lines
            if not line.lower().startswith("subject:")
        ).strip()

        print("\nSending email...")

        send_email(
            recipient,
            subject,
            body
        )

    else:
        print("\nEmail not sent.")


# ==========================================================
# LOAD MODEL AND VECTORIZER
# ==========================================================

model = joblib.load("email_intent_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# ==========================================================
# USER INPUT
# ==========================================================

message = input("Enter your message: ")


# Convert message into numbers
message_vectorized = vectorizer.transform([message])


# Predict intent
prediction = model.predict(message_vectorized)
intent = prediction[0]

print("Predicted intent:", intent)


# ==========================================================
# GENERATE EMAIL
# ==========================================================

if intent == "generate_email":

    print("\nEmail Assistant:")
    print("Let's create your email.")

    purpose = input("\nWhat is the purpose of the email? ")
    recipient_name = input("Recipient name: ")
    recipient_email = input("Recipient email: ")
    details = input("What important details should I mention? ")
    tone = input("What tone do you want? (formal/polite/simple): ")

    # Retrieve relevant information from RAG
    rag_query = f"How should I write a professional university email about {purpose}?"

    rag_results = retrieve_information(rag_query)

    context = "\n\n".join(rag_results)

    # Gemini prompt
    prompt = f"""
You are a professional university email assistant.

Create an email based on the user's information.

USER INFORMATION:
Purpose: {purpose}
Recipient name: {recipient_name}
Important details: {details}
Tone: {tone}

RELEVANT INFORMATION FROM THE KNOWLEDGE BASE:
{context}

Instructions:
- Create a clear and appropriate subject.
- Address the recipient using their name.
- Use the requested tone.
- Include all important details provided by the user.
- Follow the relevant email guidelines from the knowledge base.
- Do not invent university rules or facts.
- Keep the email concise.
- Return only the final email.
- Do not use placeholders such as [Your Name], [Your Student ID], or [Course Name].
"""

    # Generate email
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    email = response.text

    # Preview and send
    preview_and_send(email, recipient_email)


# ==========================================================
# ASSIGNMENT EXTENSION
# ==========================================================

elif intent == "assignment_extension":

    print("\nEmail Assistant:")
    print("I can help you write an assignment extension request.")

    recipient_name = input("\nRecipient name: ")
    recipient_email = input("Recipient email: ")
    student_name = input("Your name: ")
    student_id = input("Your student ID: ")
    course = input("Course name/number: ")
    assignment = input("What is the assignment? ")
    days = input("How many extra days do you need? ")
    reason = input("Why do you need the extension? ")

    # Retrieve relevant information from RAG
    rag_query = "What should I include in an assignment extension request?"

    rag_results = retrieve_information(rag_query)

    context = "\n\n".join(rag_results)

    # Gemini prompt
    prompt = f"""
You are a professional university email assistant.

Create a professional assignment extension request email.

USER DETAILS:
Student name: {student_name}
Student ID: {student_id}
Course: {course}
Recipient: {recipient_name}
Assignment: {assignment}
Additional days requested: {days}
Reason: {reason}

RELEVANT KNOWLEDGE FROM RAG:
{context}

Instructions:
- Create a clear subject line.
- Address the recipient using their name.
- Clearly mention the assignment.
- Clearly mention the number of additional days.
- Include the reason provided by the student.
- Make the request polite and respectful.
- Include the student's name, student ID, and course.
- Do not invent university rules or facts.
- End with a professional closing.
- Do not use placeholders such as [Your Name], [Your Student ID], or [Your Course Name].
- Return only the final email.
"""

    # Generate email using Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    email = response.text

    # Preview and send
    preview_and_send(email, recipient_email)


# ==========================================================
# IMPROVE EMAIL
# ==========================================================

elif intent == "improve_email":

    print("\nEmail Assistant:")
    print("I can help you improve your email.")

    email_text = input("\nPaste your email here:\n")

    # Retrieve professional email guidelines using RAG
    rag_query = "What are the guidelines for writing a professional university email?"

    rag_results = retrieve_information(rag_query)

    context = "\n\n".join(rag_results)

    # Gemini prompt
    prompt = f"""
You are a professional university email assistant.

Improve the following email while keeping its original meaning.

ORIGINAL EMAIL:
{email_text}

RELEVANT EMAIL GUIDELINES FROM THE KNOWLEDGE BASE:
{context}

Instructions:
- Keep the original meaning and important information.
- Make the language professional, polite, and clear.
- Correct grammar and sentence structure.
- Add an appropriate subject if necessary.
- Use a professional greeting and closing.
- Do not invent facts or university rules.
- Return only the improved email.
"""

    # Generate improved email
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    improved_email = response.text

    # Preview and send
    preview_and_send(improved_email)


# ==========================================================
# REPLY EMAIL
# ==========================================================

elif intent == "reply_email":

    print("\nEmail Assistant:")
    print("I can help you write a reply to an email.")

    original_email = input("\nPaste the email you received:\n")
    reply_points = input("\nWhat do you want to say in your reply? ")

    # Retrieve reply-writing guidelines using RAG
    rag_query = "What are the guidelines for writing a professional email reply?"

    rag_results = retrieve_information(rag_query)

    context = "\n\n".join(rag_results)

    # Gemini prompt
    prompt = f"""
You are a professional university email assistant.

Write a professional reply to the received email.

ORIGINAL EMAIL:
{original_email}

WHAT THE STUDENT WANTS TO SAY:
{reply_points}

RELEVANT GUIDELINES FROM THE KNOWLEDGE BASE:
{context}

Instructions:
- Understand the original email before writing the reply.
- Preserve the meaning of what the student wants to say.
- Make the reply polite, professional, and clear.
- Use an appropriate subject beginning with "Re:".
- Address the sender appropriately.
- Do not invent facts or university rules.
- Keep the reply concise.
- Return only the final email.
"""

    # Generate reply
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    generated_reply = response.text

    # Preview and send
    preview_and_send(generated_reply)


# ==========================================================
# UNKNOWN INTENT
# ==========================================================

else:

    print("\nEmail Assistant:")
    print("I understand your request, but I need more information.")