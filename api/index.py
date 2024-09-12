from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gensim.downloader as api
from fastapi.middleware.cors import CORSMiddleware

# Load the pre-trained Word2Vec model
print("Loading word vectors, this may take a few minutes...")
word_vectors = api.load("word2vec-google-news-300")
print("Word vectors loaded successfully.")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    # Adjust this to restrict allowed origins in production
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variable for the secret word
secret_word = None

# Models for input validation


class SecretWord(BaseModel):
    word: str


class GuessWords(BaseModel):
    word1: str
    word2: str


def calculate_heat_score(similarity):
    """Convert cosine similarity to a 'heat' score from 0 to 10."""
    heat_score = (similarity + 1) * 5  # Normalize to 0-10
    return round(heat_score, 2)


@app.post("/api/set_secret")
async def set_secret(secret: SecretWord):
    global secret_word
    if secret.word not in word_vectors:
        raise HTTPException(
            status_code=400, detail="The word is not in the vocabulary.")
    secret_word = secret.word
    return {"message": f"Secret word '{secret_word}' is set."}


@app.post("/api/compare")
async def compare_words(guess: GuessWords):
    global secret_word
    if not secret_word:
        raise HTTPException(status_code=400, detail="Secret word not set.")

    if guess.word1 not in word_vectors or guess.word2 not in word_vectors:
        raise HTTPException(
            status_code=400, detail="One or both words are not in the vocabulary.")

    similarity1 = word_vectors.similarity(secret_word, guess.word1)
    similarity2 = word_vectors.similarity(secret_word, guess.word2)

    heat_score1 = calculate_heat_score(similarity1)
    heat_score2 = calculate_heat_score(similarity2)

    result = {
        "word1": {"word": guess.word1, "heat_score": heat_score1},
        "word2": {"word": guess.word2, "heat_score": heat_score2},
        "closer_word": guess.word1 if similarity1 >= similarity2 else guess.word2
    }

    return result
