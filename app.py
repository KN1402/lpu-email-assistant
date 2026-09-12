import streamlit as st
import joblib
import os
import certifi
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag_engine import retrieve_information
from gmail_sender import send_email


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="LPU Academic Assistant",
    page_icon="🎓",
    layout="centered"
)


# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv(".env", override=True)

# Use the standard Python CA bundle instead of the Anaconda/Windows
# certificate store that was causing the Google HTTPS certificate error.
for _var in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(_var, None)

os.environ["SSL_CERT_FILE"] = certifi.where()

api_key = os.getenv("GEMINI_API_KEY")


# ==========================================================
# CHECK GEMINI API KEY
# ==========================================================

if not api_key:
    st.error(
        "Gemini API key was not found. Please check your .env file."
    )
    st.stop()


# ==========================================================
# GEMINI CLIENT
# ==========================================================

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=60000,
        client_args={
            "verify": certifi.where()
        }
    )
)


# ==========================================================
# LOAD NLP MODEL
# ==========================================================

model = joblib.load("email_intent_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_intent" not in st.session_state:
    st.session_state.current_intent = None

if "generated_email" not in st.session_state:
    st.session_state.generated_email = None

if "recipient_email" not in st.session_state:
    st.session_state.recipient_email = ""

if "generated_recipient_email" not in st.session_state:
    st.session_state.generated_recipient_email = ""

if "edited_email" not in st.session_state:
    st.session_state.edited_email = ""

# Prevent accidental duplicate sends.
if "last_sent_signature" not in st.session_state:
    st.session_state.last_sent_signature = ""

if "last_sent_time" not in st.session_state:
    st.session_state.last_sent_time = 0.0

if "is_sending" not in st.session_state:
    st.session_state.is_sending = False


# ==========================================================
# TITLE
# ==========================================================

st.title("🎓 LPU Academic Email Assistant")

st.write(
    "Your AI assistant for creating professional academic emails."
)


# ==========================================================
# DISPLAY CHAT HISTORY
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# ==========================================================
# USER INPUT
# ==========================================================

user_input = st.chat_input(
    "What do you want help with?"
)


# ==========================================================
# PROCESS NEW USER MESSAGE
# ==========================================================

if user_input:

    # ------------------------------------------------------
    # DISPLAY USER MESSAGE
    # ------------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)


    # ------------------------------------------------------
    # NLP INTENT CLASSIFICATION
    # ------------------------------------------------------

    message_vectorized = vectorizer.transform(
        [user_input]
    )

    prediction = model.predict(
        message_vectorized
    )

    intent = prediction[0]

    st.session_state.current_intent = intent


    # ------------------------------------------------------
    # RESPONSES FOR NON-FORM INTENTS
    # ------------------------------------------------------

    if intent == "generate_email":

        assistant_response = (
            "I can help you generate a professional academic email."
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response
        })


    elif intent == "improve_email":

        assistant_response = (
            "I can help you improve your email and make it "
            "more professional and clear."
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response
        })


    elif intent == "reply_email":

        assistant_response = (
            "I can help you write a professional reply to the "
            "email you received."
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response
        })


# ==========================================================
# ASSIGNMENT EXTENSION FORM
# ==========================================================

if st.session_state.current_intent == "assignment_extension":

    st.divider()

    st.subheader("📝 Assignment Extension Request")

    st.write(
        "Please provide the following details."
    )


    # ------------------------------------------------------
    # RECIPIENT DETAILS
    # ------------------------------------------------------

    recipient_name = st.text_input(
        "Recipient name",
        key="recipient_name"
    )

    recipient_email = st.text_input(
        "Recipient email",
        key="recipient_email"
    )


    # ------------------------------------------------------
    # STUDENT DETAILS
    # ------------------------------------------------------

    student_name = st.text_input(
        "Your name",
        key="student_name"
    )

    student_id = st.text_input(
        "Your student ID",
        key="student_id"
    )


    # ------------------------------------------------------
    # ASSIGNMENT DETAILS
    # ------------------------------------------------------

    course = st.text_input(
        "Course name/number",
        key="course"
    )

    assignment = st.text_input(
        "What is the assignment?",
        key="assignment"
    )

    days = st.text_input(
        "How many extra days do you need?",
        key="days"
    )

    reason = st.text_area(
        "Why do you need the extension?",
        key="reason"
    )


    # ======================================================
    # GENERATE EMAIL
    # ======================================================

    if st.button(
        "Generate Email",
        key="generate_extension"
    ):

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        if not recipient_name:
            st.warning("Please enter the recipient name.")

        elif not recipient_email:
            st.warning("Please enter the recipient email.")

        elif not student_name:
            st.warning("Please enter your name.")

        elif not student_id:
            st.warning("Please enter your student ID.")

        elif not course:
            st.warning("Please enter the course name/number.")

        elif not assignment:
            st.warning("Please enter the assignment name.")

        elif not days:
            st.warning("Please enter the number of extra days.")

        elif not reason:
            st.warning(
                "Please enter the reason for the extension."
            )

        else:

            # --------------------------------------------------
            # RAG
            # --------------------------------------------------

            rag_query = (
                "What should I include in an assignment "
                "extension request?"
            )

            rag_results = retrieve_information(
                rag_query
            )

            context = "\n\n".join(
                rag_results
            )


            # --------------------------------------------------
            # GEMINI PROMPT
            # --------------------------------------------------

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


INSTRUCTIONS:

- Create a clear subject line.
- Address the recipient using their name.
- Clearly mention the assignment.
- Clearly mention the number of additional days.
- Include the reason provided by the student.
- Make the request polite and respectful.
- Include the student's name, student ID, and course.
- Do not invent university rules or facts.
- End with a professional closing.
- Do not use placeholders such as [Your Name],
  [Your Student ID], or [Your Course Name].
- Return only the final email.
"""


            # --------------------------------------------------
            # GEMINI
            # --------------------------------------------------

            with st.spinner(
                "Generating your email..."
            ):

                try:

                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )

                    email = response.text

                    # Save generated email
                    st.session_state.generated_email = email

                    # Save recipient separately from the input widget.
                    st.session_state.generated_recipient_email = (
                        recipient_email
                    )

                    # Put generated email into editable field
                    st.session_state.edited_email = email

                    st.success(
                        "✅ Email generated successfully!"
                    )

                except Exception as e:

                    st.error(
                        "Unable to generate the email right now."
                    )

                    st.code(str(e))


# ==========================================================
# DISPLAY GENERATED EMAIL
# ==========================================================

if st.session_state.generated_email:

    st.divider()

    st.subheader("📧 Generated Email")

    # ------------------------------------------------------
    # EDITABLE EMAIL
    # ------------------------------------------------------

    edited_email = st.text_area(
        "Email Preview - You can edit this before sending",
        key="edited_email",
        height=350
    )


    # ======================================================
    # SEND EMAIL
    # ======================================================

    if st.button(
        "📤 Send Email",
        key="send_generated_email",
        disabled=st.session_state.is_sending
    ):

        recipient = st.session_state.generated_recipient_email

        # --------------------------------------------------
        # CHECK RECIPIENT
        # --------------------------------------------------

        if not recipient:

            st.error(
                "Recipient email is missing."
            )

        else:

            # --------------------------------------------------
            # EXTRACT SUBJECT
            # --------------------------------------------------

            lines = edited_email.splitlines()

            subject = ""
            body_start = 0

            for i, line in enumerate(lines):

                if line.strip().lower().startswith("subject:"):

                    subject = line.split(
                        ":",
                        1
                    )[1].strip()

                    body_start = i + 1

                    while (
                        body_start < len(lines)
                        and not lines[body_start].strip()
                    ):
                        body_start += 1

                    break

            # --------------------------------------------------
            # IF NO SUBJECT FOUND
            # --------------------------------------------------

            if not subject:

                subject = "Academic Email"
                body = edited_email.strip()

            else:

                body = "\n".join(
                    lines[body_start:]
                ).strip()

            # --------------------------------------------------
            # PREVENT ACCIDENTAL DUPLICATE SENDS
            # --------------------------------------------------

            send_signature = (
                recipient.strip().lower()
                + "\n"
                + subject.strip().lower()
                + "\n"
                + body.strip()
            )

            now = time.time()

            if (
                send_signature == st.session_state.last_sent_signature
                and now - st.session_state.last_sent_time < 60
            ):

                st.warning(
                    "⚠️ This exact email was already sent. "
                    "Duplicate send blocked for 60 seconds."
                )

            else:

                # Mark sending immediately so the button becomes disabled
                # on the next Streamlit rerun.
                st.session_state.is_sending = True

                try:

                    with st.spinner(
                        "Sending email... Please wait."
                    ):

                        send_email(
                            recipient,
                            subject,
                            body
                        )

                    # Remember this exact email after successful send.
                    st.session_state.last_sent_signature = send_signature
                    st.session_state.last_sent_time = time.time()

                    st.success(
                        f"✅ Email sent successfully to {recipient}"
                    )

                except Exception as e:

                    st.error(
                        "❌ Failed to send the email."
                    )

                    st.code(str(e))

                finally:

                    st.session_state.is_sending = False
