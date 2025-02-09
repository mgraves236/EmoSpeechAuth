import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import random
from sklearn.utils import shuffle
from augmentation import AddGaussianNoise, EmbeddingScale, EmbeddingDropout
from config import emo_model_name, sv_model_name, rd_seed


class AudioDataset(Dataset):
    def __init__(self, speaker_dirs, root_dir, samplerate=16000, augment_prob=0.5):
        self.root_dir = root_dir
        self.target_sample_rate = samplerate
        if 'audio_speech_actors_01-24' in speaker_dirs:
            speaker_dirs.remove('audio_speech_actors_01-24')

        self.augment_prob = augment_prob

        self.augmentations = [
            AddGaussianNoise(),
            EmbeddingDropout(),
            EmbeddingScale()
        ]

        self.file_paths = []
        self.labels = []

        for label, speaker in enumerate(speaker_dirs):
            speaker_dir = os.path.join(root_dir, speaker)
            for file_name in os.listdir(speaker_dir):
                file_path = os.path.join(speaker_dir, file_name)
                self.file_paths.append(file_path)
                self.labels.append(label)

        self.file_paths, self.labels = shuffle(self.file_paths, self.labels, random_state= rd_seed)
        self.label_to_indices = self.__create_label_index_map__()

    def __len__(self):
        return len(self.file_paths)

    def __create_label_index_map__(self):
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def __getmodelpath__(self, model_name, original_path):
        parts = original_path.split('/')
        if len(parts) > 3:
            parts[2] = f"{parts[2].split('-')[0]}-{model_name}"

        return '/'.join(parts)

    def __getitem__(self, idx):

        anchor_path = self.file_paths[idx]
        anchor_label = self.labels[idx]

        is_positive = random.choice([True, False])

        if is_positive:
            positive_idx = random.choice(self.label_to_indices[anchor_label])
            while positive_idx == idx:
                positive_idx = random.choice(self.label_to_indices[anchor_label])
            pair_path = self.file_paths[positive_idx]
            label = 0
        else:
            negative_label = random.choice(
                [lbl for lbl in self.label_to_indices.keys() if lbl != anchor_label]
            )
            negative_idx = random.choice(self.label_to_indices[negative_label])
            pair_path = self.file_paths[negative_idx]
            label = 1

        emo_embd_anchor = torch.tensor(np.load(anchor_path), dtype=torch.float32)
        emo_embd_pair = torch.tensor(np.load(pair_path), dtype=torch.float32)

        sv_anchor_path = self.__getmodelpath__(sv_model_name, anchor_path)
        sv_embd_anchor = torch.tensor(np.load(sv_anchor_path), dtype=torch.float32)

        sv_pair_path = self.__getmodelpath__(sv_model_name, pair_path)
        sv_embd_pair = torch.tensor(np.load(sv_pair_path), dtype=torch.float32)

        return emo_embd_anchor, sv_embd_anchor, emo_embd_pair, sv_embd_pair, label


def split_speakers(root_dir, train_ratio=0.7, val_ratio=0.15, seed=rd_seed):
    random.seed(seed)
    speakers = sorted(os.listdir(root_dir))
    random.shuffle(speakers)
    total_speakers = len(speakers)
    train_end_idx = int(total_speakers * train_ratio)
    val_end_idx = train_end_idx + int(total_speakers * val_ratio)

    train_speakers = speakers[:train_end_idx]
    val_speakers = speakers[train_end_idx:val_end_idx]
    test_speakers = speakers[val_end_idx:]

    return train_speakers, val_speakers, test_speakers
