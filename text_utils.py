# text_utils.py
import re

def clean_text(text):
    """
    Cleans the input text by:
    1. Ensuring the input is a string.
    2. Converting to lowercase.
    3. Removing URLs.
    4. Removing non-alphabetic characters (keeps only a-z and spaces).
    5. Removing short words (1-2 characters).
    6. Removing extra spaces.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-alphabetic characters
    text = re.sub(r"\b\w{1,2}\b", "", text)  # Remove short words
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# You can add other text processing related functions here in the future if needed