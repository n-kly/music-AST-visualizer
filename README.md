# Audio Embeddings Visualizer

![Demo Video](link-to-demo-video)

## Overview

This project is a data visualization tool for music embeddings that lets you explore the relationships between different songs, artists, and genres. It converts raw audio files into deep audio embeddings and then visualizes these embeddings to show how similar or different the songs are from each other.

We source our visualisation data by scraping songs from YouTube music and fetching metadata and preview links from Spotify; the raw audio from this is first preprocessed to extract audio features by converting the waveforms into mel spectrograms, which are a type of visual representation of sound. Traditional computer vision techniques can now be applied here to learn patterns and representations of the audio by passing the mel spectrograms into a pre-trained audio transformer encoder model such as [AST](https://arxiv.org/abs/2007.14062). The dimensionality of these embeddings can then be reduced from 768 to 2 by using principal component analysis to map the embeddings into a 2D space, at which point a k-means clustering algorithm is used to identify groups of similar sounds.  

However, we found that there were few pre-trained available models that were specialized for musical embeddings, so we decided to take a shot at making our own. We created a simple 2.4M parameter audio spectrogram transformer taking inspiration from state-of-the-art architectures trained on ~200 hours of music from this [dataset](https://huggingface.co/datasets/lewtun/music_genres). For this small a model, the performance for this task was remarkable, being able to identify general audio patterns to differentiate between most music types and genres; however, due to limitations in our dataset size/ quality and limited compute, the model has a lot of room for improvement.


## Model Overview

### Preprocessing

The audio preprocessing pipeline is a crucial part of this project. It converts raw waveforms into mel spectrograms, breaks them into chunks, and divides them further into patches.

- **Waveform to Mel Spectrogram**: Converts audio signals into 128 Mel Spectrogram features [Understanding the Mel Spectrogram](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53).
- **Chunking**: Divides the spectrogram into overlapping chunks of 1024 frames each; each chunk is treated as a separate audio file for inference, these chunks are mean pooled.
- **Patching**: Splits each chunk into overlapping 32x32 patches; each patch is treated as a "token" for the transformer

Here's an overview of the `AudioFeatureExtractor` class methods:

#### Methods

- **`__init__`**: Initializes the feature extractor with specified parameters.
  - Parameters: `sample_rate`, `n_mels`, `window_size`, `hop_size`, `chunk_size`, `chunk_overlap`, `patch_size`, `patch_overlap`, `do_normalize`, `mean`, `std`
  - Returns: None
  - Dimensions: None

- **`__call__`**: Converts the audio file at `file_path` into normalized mel spectrogram patches.
  - Parameters: `file_path`, `tokenize`
  - Returns: List of patches
  - Dimensions: `128 x t` for mel spectrogram, `128 x 1024` for chunks, `32 x 32` for patches

- **`calculate_dataset_stats`**: Computes mean and standard deviation for the dataset.
  - Parameters: `file_paths`, `extractor`
  - Returns: Mean, Standard Deviation
  - Dimensions: None

- **`chunk_spectrogram`**: Splits the spectrogram into chunks.
  - Parameters: `spectrogram`, `chunk_size`, `chunk_overlap`
  - Returns: Array of chunks
  - Dimensions: `128 x 1024`

- **`create_patches`**: Creates patches from a chunk.
  - Parameters: `chunk`
  - Returns: Array of patches
  - Dimensions: `32 x 32`

### Model Architecture

The model is a custom transformer encoder designed to process audio embeddings. Below is a brief overview of its architecture:

1. **Convolution Layer**: Reduces dimensionality of 32x32 patches to 16x16.
2. **Flatten and Linear Projection**: Converts the 16x16 embedding into a 768-dimensional vector.
3. **Positional Embedding**: Adds positional encoding to the embeddings.
4. **Transformer Encoder**: Processes the embeddings with multiple layers of transformer encoders.
5. **Output Layer**: Projects the output to a 768-dimensional vector.

You can find the pretrained model in /model/audio_embedding_model.pth

#### Using the Pretrained Model for local inference

```python
import torch
from model.audio_preprocessor import AudioFeatureExtractor
from model.custom_model import AudioTransformerModel

# Initialize audio feature extractor
feature_extractor = AudioFeatureExtractor()

# Load the pretrained model
model = AudioTransformerModel(patch_size=32, num_layers=12, num_heads=8, d_model=768, dim_feedforward=2048)
model.load_state_dict(torch.load('audio_embedding_model.pth'))
model.eval()

# Generate custom embedding
def generate_custom_embedding(audio_path):
    inputs = feature_extractor(audio_path)
    inputs = torch.tensor(inputs).unsqueeze(0)
    embeddings = []
    
    for chunk in inputs:
        with torch.no_grad():
            chunk_embedding = model(chunk)
        embeddings.append(chunk_embedding.mean(dim=0))
    
    embeddings = torch.stack(embeddings)
    final_embedding = embeddings.mean(dim=0)
    
    return final_embedding.cpu().numpy()
```

## Visualizer usage

To install the necessary packages and set up the environment, follow these steps:

```sh
pip install -r requirements.txt
```

Run the Dash app:

```sh
python app.py
```

## Next Steps
Overall, while the project turned out great, there are definitely many areas for improvement/ development; here are some ideas we came up with along the way:

- **UI Improvements**:
  - Add a way for users to upload their own playlists/ songs
- **Visualization Improvements**:
  - Add information panels for artists and genres similar to how the songs work.
  - Plot songs atop known genre clusters using k-means cluster centres.
- **Model Enhancements**:
  - Increase model parameter size/ Increase the size and diversity of the training dataset.
  - Find/ manually create a small sample of labelled data to use for portions of supervised learning.
  - Experiment with different model architectures/pre-trained models (e.g., LTU).
  - Develop an enhanced evaluation metric for this specific task that would better allow you to compare model performances.
  - Loss function was iffy the entire way through; take a second look and try to implement contrastive loss.
- **Data Acquisition**:
  - Due to Spotify API not being allowed for AI/ML use, we had to generate embeddings and train the model using YouTube music, which is very slow and the quality of data isn't as high (Intros, Outros, Music videos instead of raw audio, etc.). We used an external data set to increase training data, but finding an efficient in-house solution would be helpful.
  - Our model was trained exclusively on music but adding small amounts of other audio patterns could allow the model to generalize better  
  - Filter for more diverse and representative songs. Many songs were from similar genres, e.g., latin, latino, and reggaeton, which skewed the representation.

## Citations
- Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, & Ilya Sutskever. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. [Whisper](https://github.com/openai/whisper)
- Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, & Michael Auli. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. [Wave2Vec](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2)
- Yuan Gong, Hongyin Luo, Alexander H. Liu, Leonid Karlinsky, & James Glass. (2024). Listen, Think, and Understand. [LTU](https://github.com/YuanGongND/ltu)
- Yuan Gong, Yu-An Chung, & James Glass (2021). AST: Audio Spectrogram Transformer. In Proc. Interspeech 2021 (pp. 571–575). [AST](https://github.com/YuanGongND/ast)
