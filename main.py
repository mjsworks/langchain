import os

from dotenv import load_dotenv

load_dotenv()

opanai_api_key = os.environ.get("OPENAI_API_KEY")

if opanai_api_key is not None:
    print(f"openai_api_key is set to: {opanai_api_key}")
else:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")
