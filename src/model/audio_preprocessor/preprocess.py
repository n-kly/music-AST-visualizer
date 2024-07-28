import torch
import torchaudio
import numpy as np
from tqdm import tqdm

class AudioFeatureExtractor:
    def __init__(
        self,
        sample_rate=16000,
        n_mels=128,
        window_size=25,
        hop_size=10,
        chunk_size=512,
        do_normalize=True,
        mean=None,
        std=None
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.window_size = window_size
        self.hop_size = hop_size
        self.chunk_size = chunk_size
        self.do_normalize = do_normalize
        
        self.mean = mean
        self.std = std

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            win_length=int(window_size / 1000 * sample_rate),
            hop_length=int(hop_size / 1000 * sample_rate)
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, file_path):
        waveform, original_sample_rate = torchaudio.load(file_path)
        if original_sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)(waveform)
        
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

        mel_spec = self.mel_spec(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        features = log_mel_spec.squeeze().numpy()

        # # Pad or truncate to chunk_size
        # if features.shape[1] < self.chunk_size:
        #     pad_width = ((0, 0), (0, self.chunk_size - features.shape[1]))
        #     features = np.pad(features, pad_width, mode='constant')
        # elif features.shape[1] > self.chunk_size:
        #     features = features[:, :self.chunk_size]

        # Normalize if required
        if self.do_normalize:
            if self.mean is None or self.std is None:
                self.mean = np.mean(features)
                self.std = np.std(features)
            features = (features - self.mean) / (self.std + 1e-6)  # Add small epsilon to avoid division by zero

        # Chunk the spectrogram
        chunks = self.chunk_spectrogram(features, chunk_size=self.chunk_size, hop_size=self.chunk_size/4)

        return chunks

    @staticmethod
    def calculate_dataset_stats(file_paths, extractor):
        all_features = []
        for file_path in tqdm(file_paths, desc="Calculating dataset mean/std"):
            features = extractor(file_path)
            all_features.append(features)
        all_features = np.concatenate(all_features, axis=0)
        mean = np.mean(all_features)
        std = np.std(all_features)
        return mean, std
    
    @staticmethod
    def chunk_spectrogram(spectrogram, chunk_size=512, hop_size=256):
        chunks = []
        for i in range(0, spectrogram.shape[1] - chunk_size + 1, int(hop_size)):
            chunk = spectrogram[:, i:i + chunk_size]
            chunks.append(chunk)
        return np.array(chunks)