# Audio Embeddings Visualizer

![Demo Video](link-to-demo-video)

## Overview

This project is a comprehensive audio embeddings visualizer designed to analyze and visualize the embedding space of songs. The system processes raw audio waveforms into mel spectrograms, feeds the data into a custom transformer encoder model, and visualizes the embeddings using PCA and k-means clustering. The data is scraped from YouTube, with metadata and previews provided by Spotify. The visualizations are created using Dash and Plotly, offering an interactive and insightful way to explore the audio embedding space.

## Model Overview

### Preprocessing

The audio preprocessing pipeline is a crucial part of this project. It converts raw waveforms into mel spectrograms, breaks them into chunks, and further divides them into patches.

- **Waveform to Mel Spectrogram**: Converts audio signals into 128 mel spectrogram features.
- **Chunking**: Divides the spectrogram into overlapping chunks of 1024 frames each.
- **Patching**: Splits each chunk into overlapping 32x32 patches.

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

#### Using the Pretrained Model

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
Overall, while the project turned out great, there are definitely a lot of areas for improvement/ development, here are some we thought up along the way.

- **Visualization Improvements**:
  - Add information panels for artists and genres similar to how the songs work.
  - Plot songs atop known genre clusters using k-means cluster centers.
- **Model Enhancements**:
  - Increase model parameter size.
  - Increase the size and diversity of the training dataset.
  - Experiment with different model architectures/ pretrained models (e.g., LTU).
  - Loss function was iffy the entire way through, take a second look and try implement contrastive loss.
- **Data Acquisition**:
  - Due to Spotify API not being allowed for AI/ML use we had to generate embeddings/ train the model using YouTube music which is very slow and quality of data isn't as high (Intros, Outros, Music videos instead of raw audio, etc.).
  - Filter for more diverse and representative songs, many songs were from similar genres e.g. latin, latino, reggaeton which skewed the representation.

## Citations

- [Wave2Vec](https://arxiv.org/abs/1904.05862)
- [Whisper](https://openai.com/research/whisper)
- [BERT](https://arxiv.org/abs/1810.04805)
- [AST](https://arxiv.org/abs/2007.14062)
- [LTU](https://arxiv.org/abs/1906.00295)



