import os
import base64

from email.message import EmailMessage

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def get_gmail_service():

    creds = None

    # Check if we already have a saved login
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file(
            "token.json",
            SCOPES
        )

    # If login is missing or expired
    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES
            )

            creds = flow.run_local_server(port=0)

        # Save login information
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    # Create Gmail API service
    service = build(
        "gmail",
        "v1",
        credentials=creds
    )

    return service


def send_email(to_email, subject, body):

    service = get_gmail_service()

    message = EmailMessage()

    message.set_content(body)

    message["To"] = to_email
    message["Subject"] = subject

    # Encode email for Gmail API
    encoded_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode()

    create_message = {
        "raw": encoded_message
    }

    # Send email
    send_message = service.users().messages().send(
        userId="me",
        body=create_message
    ).execute()

    print("Email sent successfully!")
    print("Message ID:", send_message["id"])