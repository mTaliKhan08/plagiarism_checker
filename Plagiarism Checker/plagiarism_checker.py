import spacy
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# Load the spaCy model
nlp = spacy.load("en_core_web_md")


def load_text(file_path):
    """Load text from a file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def preprocess_text(text):
    """Preprocess the text by tokenizing and removing stop words"""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)


def calculate_similarity(text1, text2):
    """Calculate similarity between two texts"""
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)


def main():
    print("Welcome to the Plagiarism Detector")
    file_path1 = input("Enter the path to the first document: ").strip()
    file_path2 = input("Enter the path to the second document: ").strip()

    text1 = load_text(file_path1)
    text2 = load_text(file_path2)

    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)

    similarity = calculate_similarity(processed_text1, processed_text2)
    print(f"Similarity: {similarity * 100:.2f}%")


if __name__ == "__main__":
    main()
