"""Quick manual check that settings load and validate."""

from pydantic import ValidationError

from app.config import Settings, get_settings

settings = get_settings()
print("Loaded settings:")
print(settings)
print("Key starts with:", settings.gemini_api_key.get_secret_value()[:6] + "...")

# Fail-fast check: this SHOULD raise.
try:
    Settings(chunk_size=200, chunk_overlap=300)
except ValidationError as e:
    print("\nBad config caught ✅")
    print(e.errors()[0]["msg"])