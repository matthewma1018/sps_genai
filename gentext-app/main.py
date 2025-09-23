from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.embedding import Embedding

app = FastAPI()
# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)
embedding_model = Embedding()

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class WordEmbeddingRequest(BaseModel):
    word: str

class SentenceEmbeddingsRequest(BaseModel):
    query: str
    info: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/word_embedding")
def get_word_embedding(request: WordEmbeddingRequest):
    embedding = embedding_model.word_embedding(request.word)
    return {"message": "First 10 elements of the embedding vector:", "embedding": embedding}

@app.post("/sentence_embeddings")
def get_sentence_embeddings(request: SentenceEmbeddingsRequest):
    embeddings = embedding_model.sentence_embeddings(request.query, request.info)
    return {"similarity": embeddings}