import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Définir le chemin du cache pour le modèle
cache_dir = os.path.join(os.getcwd(), "model_cache")

app = FastAPI()

# Charger le modèle BlenderBot depuis le cache local
try:
    model_id = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir)
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du modèle : {str(e)}")

class TextGenerationRequest(BaseModel):
    prompt: str

@app.post("/generate/")
def generate_text(request: TextGenerationRequest):
    try:
        # Tokeniser le prompt
        inputs = tokenizer(request.prompt, return_tensors="pt")
        
        # Générer la réponse avec des paramètres optimisés
        reply_ids = model.generate(
            **inputs,
            max_length=150,  # Augmenter la longueur maximale
            num_beams=5,  # Beam search pour améliorer la qualité de la réponse
            temperature=0.7,  # Contrôler la créativité du modèle
            top_k=50,  # Filtrage des top-k tokens
            top_p=0.9,  # Nucleus sampling
            repetition_penalty=1.2  # Pénaliser les répétitions
        )
        
        # Décoder la réponse générée
        response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        return {"generated_text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")
