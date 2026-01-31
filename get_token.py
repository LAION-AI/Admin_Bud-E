from google.oauth2 import service_account
from google.auth.transport.requests import Request

SA_PATH = "./gemini-school-471209-3972614b5ec6.json" 

# Create credentials from service account JSON
creds = service_account.Credentials.from_service_account_file(
    SA_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

# Force refresh to get a fresh access token
creds.refresh(Request())

print(creds.token)
