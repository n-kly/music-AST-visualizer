import os
import json
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTModel
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

# Load metadata
with open('metadata/songs_metadata.json', 'r') as f:
    all_songs = json.load(f)

# Initialize AST model and feature extractor
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# Initialize Pinecone
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')  # Make sure this is set in your .env file
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_name = 'music-embeddings'

# Check if index exists, if not create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # AST embedding dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Choose an appropriate region
        )
    )

index = pc.Index(index_name)

def generate_embedding(model, audio_path):
    waveform, original_sample_rate = torchaudio.load(audio_path)
    waveform = waveform.squeeze().numpy()

    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu')for k, v in inputs.items()}  # Move inputs to GPU
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use mean pooling to get a single vector
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Move back to CPU for further processing
    embeddings = embeddings.mean(axis=0)
    return embeddings

def process_song(song):
    audio_path = os.path.join('previews', f"{song['id']}.mp3")
    if os.path.exists(audio_path):
        embedding = generate_embedding(audio_path)
        
        # Store in Pinecone
        index.upsert(vectors=[(song['id'], embedding.tolist(), {
            'name': song['name'],
            'artist': song['artists'][0]['name'],
            'album': song['album']['name'],
            'genre': song.get('genre', 'Unknown')
        })])

def main():
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_song, song) for song in all_songs]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating embeddings"):
            future.result()  # Raise exceptions if any

    print("All embeddings generated and stored in Pinecone.")

if __name__ == "__main__":
    main()
