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
        chunk_size=1024,
        chunk_overlap=256,
        patch_size=32,
        patch_overlap=8,
        mean=None,
        std=None
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.window_size = window_size
        self.hop_size = hop_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        
        self.mean = mean
        self.std = std

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            win_length=int(self.window_size / 1000 * self.sample_rate),
            hop_length=int(self.hop_size / 1000 * self.sample_rate)
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, file_path, do_chunk=True, do_patch=True, do_normalize=True):
        waveform, original_sample_rate = torchaudio.load(file_path)
        if original_sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)(waveform)
        
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

        mel_spec = self.mel_spec(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)    
        features = log_mel_spec.squeeze().numpy()
        
        if do_normalize and self.mean is not None and self.std is not None:
            features = (features - self.mean) / (self.std + 1e-6)
        
        if(do_chunk):
            chunks = self.chunk_spectrogram(features, self.chunk_size, self.chunk_overlap)
            if(do_patch):
                patches = [self.create_patches(chunk, self.patch_size, self.patch_overlap) for chunk in chunks]
                return patches
            return chunks
        return features

    @staticmethod
    def calculate_dataset_stats(file_paths, extractor):
        all_features = []
        for file_path in tqdm(file_paths, desc="Calculating dataset mean/std"):
            features = extractor(file_path, do_patch=False, do_normalize=False)
            all_features.extend(features)
        all_features = np.concatenate(all_features, axis=0)
        mean = np.mean(all_features)
        std = np.std(all_features)
        return mean, std
    
    @staticmethod
    def chunk_spectrogram(spectrogram, chunk_size=1024, chunk_overlap=256):
        chunks = []
        for i in range(0, spectrogram.shape[1], int(chunk_size)-chunk_overlap):
            # Pad chunk size if the last chunk is smaller than chunk_size
            if i + chunk_size > spectrogram.shape[1]:
                pad = chunk_size - (spectrogram.shape[1] - i)
                chunk = np.pad(spectrogram[:, i:], ((0, 0), (0, pad)), mode='constant')

            else:
                chunk = spectrogram[:, i:i + chunk_size]
            
            chunks.append(chunk)
        return np.array(chunks)

    @staticmethod    
    def create_patches(chunk, patch_size=32, patch_overlap=8):
        patches = []
        for i in range(0, chunk.shape[0], patch_size - patch_overlap):
            for j in range(0, chunk.shape[1], patch_size - patch_overlap):
                # Pad patch size if the last patch is smaller than patch_size
                if i + patch_size > chunk.shape[0] and j + patch_size > chunk.shape[1]:
                    pad_i = patch_size - (chunk.shape[0] - i)
                    pad_j = patch_size - (chunk.shape[1] - j)
                    patch = np.pad(chunk[i:, j:], ((0, pad_i), (0, pad_j)), mode='constant')

                elif i + patch_size > chunk.shape[0]:
                    pad = patch_size - (chunk.shape[0] - i)
                    patch = np.pad(chunk[i:, j:j + patch_size], ((0, pad), (0, 0)), mode='constant')
                
                elif j + patch_size > chunk.shape[1]:
                    pad = patch_size - (chunk.shape[1] - j)
                    patch = np.pad(chunk[i:i + patch_size, j:], ((0, 0), (0, pad)), mode='constant')
                
                else:
                    patch = chunk[i:i + patch_size, j:j + patch_size]
                patches.append(patch)
        return np.array(patches)