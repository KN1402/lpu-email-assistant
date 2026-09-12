
import os
import json
import base64

import streamlit as st

from email.message import EmailMessage

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def get_gmail_service():

    creds = None

    # -----------------------------------------
    # 1. LOCAL: use token.json if it exists
    # -----------------------------------------
    if os.path.exists("token.json"):

        creds = Credentials.from_authorized_user_file(
            "token.json",
            SCOPES
        )

    # -----------------------------------------
    # 2. STREAMLIT CLOUD: use token from Secrets
    # -----------------------------------------
    elif "GMAIL_TOKEN_JSON" in st.secrets:

        token_data = json.loads(
            st.secrets["GMAIL_TOKEN_JSON"]
        )

        creds = Credentials.from_authorized_user_info(
            token_data,
            SCOPES
        )

    # -----------------------------------------
    # 3. Refresh / create credentials
    # -----------------------------------------
    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:

            creds.refresh(Request())

        else:

            # LOCAL ONLY
            if os.path.exists("credentials.json"):

                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json",
                    SCOPES
                )

                creds = flow.run_local_server(port=0)

            else:
                raise Exception(
                    "Gmail credentials are not configured on Streamlit Cloud."
                )

        # Save token locally only
        if os.path.exists("credentials.json"):

            with open("token.json", "w") as token:
                token.write(creds.to_json())

    # -----------------------------------------
    # 4. Create Gmail API service
    # -----------------------------------------
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

    encoded_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode()

    create_message = {
        "raw": encoded_message
    }

    send_message = service.users().messages().send(
        userId="me",
        body=create_message
    ).execute()

    print("Email sent successfully!")
    print("Message ID:", send_message["id"])