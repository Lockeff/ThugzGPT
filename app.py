from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

# Remplacez par votre propre jeton d'API Hugging Face
HUGGINGFACE_API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"

headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
}

class TextGenerationRequest(BaseModel):
    prompt: str

@app.post("/generate/")
def generate_text(request: TextGenerationRequest):
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": request.prompt})
        response.raise_for_status()  # Vérifie que la requête a réussi
        generated_text = response.json()[0]["generated_text"]
        return {"generated_text": generated_text}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")
