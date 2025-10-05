import torch
import torch.nn as nn
import os
import torchaudio
import numpy as np
import torch.optim as optim
import tqdm
import os
import gc
import torch.nn.functional as F
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Choose upstream model (frontend) for emotion embeddings
emo_model_array = ["emotion2vec", "wav2vec"]

# Choose upstream model (frontend) for speaker verification embeddings

sv_model_array = ["ecapa2", "ecapa-tdnn", "resnet"]

emo_model_name = emo_model_array[0]
sv_model_name = sv_model_array[0]


root_dir = '' + "-" + sv_model_name
classes = sorted(os.listdir(root_dir))

output_dir = 'EmoSpeechAuth_' + sv_model_name + '_' + emo_model_name
os.makedirs(output_dir, exist_ok=True)


class EmoSpeechAuth(nn.Module):
    def __init__(self, emo_model_name, sv_model_name):
        super(EmoSpeechAuth, self).__init__()
        self.emo_model_name = emo_model_name
        self.sv_model_name = sv_model_name

        print(emo_model_name, sv_model_name)

        if emo_model_name == "emotion2vec":
            self.emo_size = 768
        elif emo_model_name == 'wav2vec':
            self.emo_size = 1024  # T x 1024
        # Speaker verification model
        if sv_model_name == "ecapa-tdnn" or sv_model_name == "ecapa2":
            self.sv_size = 192
        if sv_model_name == "resnet":
            self.sv_size = 256

        # Linear classifier crossattention
        self.fc0 = nn.Linear(self.sv_size, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)

        self.emo_projection = nn.Linear(self.emo_size, self.sv_size)

        self.cross_attention = nn.MultiheadAttention(embed_dim=self.sv_size // 4, num_heads=1, batch_first=True,
                                                     dropout=0.4)

        self.relu = nn.Mish()

        self.dropout = nn.Dropout(0.4)

    def flow_network(self, input):
        x = input
        x = self.fc0(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def forward(self, emo_embd1, sv_embd1):

        emo_embd1 = F.normalize(emo_embd1, p=2, dim=1)
        sv_embd1 = F.normalize(sv_embd1, p=2, dim=1)

        # Cross-attention
        # Q, K, V
        emo_embd1 = self.emo_projection(emo_embd1)

        emo_embd1 = emo_embd1.view(emo_embd1.shape[0], 4, self.sv_size // 4)
        sv_embd1 = sv_embd1.view(emo_embd1.shape[0], 4, self.sv_size // 4)

        embd1, attn_output_weights1 = self.cross_attention(sv_embd1, emo_embd1, emo_embd1)
        embd1 = embd1.reshape(emo_embd1.shape[0], self.sv_size)

        x1 = self.flow_network(embd1)

        return x1


def __getmodelpath__(model_name, original_path):
    parts = original_path.split('/')
    if len(parts) > 4:
        parts[3] = f"{parts[3].split('-')[0]}-{model_name}"

    return '/'.join(parts)


model = EmoSpeechAuth(emo_model_name, sv_model_name)
model.to(device)
model.load_state_dict(torch.load('e2_e2v_seed0.pth')['model_state_dict'])
model.eval()

# List files as an array
for class_name in classes:
    print(class_name)
    class_dir = os.path.join(root_dir, class_name)
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)

        class_dir_output = os.path.join(output_dir, class_name)
        os.makedirs(class_dir_output, exist_ok=True)

        sv_embd = torch.tensor(np.load(file_path), dtype=torch.float32)
        emo_embd = torch.tensor(np.load(__getmodelpath__(emo_model_name, file_path)), dtype=torch.float32)

        embd = model(emo_embd.unsqueeze(0).to(device), sv_embd.unsqueeze(0).to(device))
        embd = embd.squeeze(0)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        np.save(os.path.join(class_dir_output, f"{file_name}.npy"), embd.detach().cpu().numpy())
        # Save embeddings as [embd_dim]





