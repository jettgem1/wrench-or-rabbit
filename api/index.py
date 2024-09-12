from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import spacy
from typing import Optional

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
nlp: Optional[spacy.language.Language] = None
secret_word = None

# Models for input validation


class SecretWord(BaseModel):
    word: str


class GuessWords(BaseModel):
    word1: str
    word2: str


def load_model():
    """Load the Spacy model if not already loaded."""
    global nlp
    if nlp is None:
        print("Loading Spacy model...")
        nlp = spacy.load("en_core_web_md")  # Use a smaller Spacy model
        print("Spacy model loaded.")


def calculate_similarity(word1, word2):
    """Calculate cosine similarity using Spacy vectors."""
    vec1 = nlp(word1).vector
    vec2 = nlp(word2).vector
    return vec1.dot(vec2) / (vec1.norm() * vec2.norm())


def calculate_heat_score(similarity):
    """Convert cosine similarity to a 'heat' score from 0 to 10."""
    heat_score = (similarity + 1) * 5  # Normalize to 0-10
    return round(heat_score, 2)


@app.post("/api/set_secret")
async def set_secret(secret: SecretWord):
    global secret_word
    load_model()  # Load model on demand
    secret_word = secret.word
    return {"message": f"Secret word '{secret_word}' is set."}


@app.post("/api/compare")
async def compare_words(guess: GuessWords):
    global secret_word
    load_model()  # Load model on demand

    if not secret_word:
        raise HTTPException(status_code=400, detail="Secret word not set.")

    similarity1 = calculate_similarity(secret_word, guess.word1)
    similarity2 = calculate_similarity(secret_word, guess.word2)

    heat_score1 = calculate_heat_score(similarity1)
    heat_score2 = calculate_heat_score(similarity2)

    result = {
        "word1": {"word": guess.word1, "heat_score": heat_score1},
        "word2": {"word": guess.word2, "heat_score": heat_score2},
        "closer_word": guess.word1 if similarity1 >= similarity2 else guess.word2
    }

    return result
