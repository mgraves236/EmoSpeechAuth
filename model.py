import torch
import torch.nn as nn
import torch.nn.functional as F
from config import version, dropout_v


class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, scale=10.0):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, output1, output2, label):
        output1 = output1 / torch.norm(output1, dim=1, keepdim=True)
        output2 = output2 / torch.norm(output2, dim=1, keepdim=True)

        cosine_sim = F.cosine_similarity(output1, output2).double()
        cosine_distance = self.scale * (1 - cosine_sim + 1e-8).double()
        loss = torch.mean(
            (1 - label) * torch.pow(cosine_distance, 2) +  # Similar pairs (label = 0)
            (label) * torch.pow(torch.clamp(- cosine_distance + self.margin, min=0.0), 2)
            # Dissimilar pairs (label = 1)
        )
        return loss, cosine_distance


class EmoSpeechAuth(nn.Module):
    def __init__(self, emo_model_name, sv_model_name):
        super(EmoSpeechAuth, self).__init__()
        self.emo_model_name = emo_model_name
        self.sv_model_name = sv_model_name

        if emo_model_name == "emotion2vec":
            self.emo_size = 768
        elif emo_model_name == 'wav2vec':
            self.emo_size = 1024  # T x 1024
        # Speakver verification model
        if sv_model_name == "ecapa-tdnn" or sv_model_name == "ecapa2":
            self.sv_size = 192
        if sv_model_name == "resnet":
            self.sv_size = 256

        # Linear classifier concat
        if version == 2:
            self.fc0 = nn.Linear(self.sv_size + self.emo_size, 1024)
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)

        # Linear classifier crossattention
        if version == 0:
            self.fc0 = nn.Linear(self.sv_size, 1024)
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)

        self.emo_projection = nn.Linear(self.emo_size, self.sv_size)

        self.cross_attention = nn.MultiheadAttention(embed_dim=self.sv_size // 4, num_heads=1, batch_first=True,
                                                     dropout=dropout_v)

        if version == 0 or version == 2:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Mish()

        self.dropout = nn.Dropout(dropout_v)

    def flow_network(self, input):
        x = input
        x = self.fc0(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def forward(self, emo_embd1, sv_embd1, emo_embd2, sv_embd2):

        emo_embd1 = F.normalize(emo_embd1, p=2, dim=1)
        sv_embd1 = F.normalize(sv_embd1, p=2, dim=1)

        emo_embd2 = F.normalize(emo_embd2, p=2, dim=1)
        sv_embd2 = F.normalize(sv_embd2, p=2, dim=1)

        # Concatentation
        if version == 2:
            embd1 = torch.cat((emo_embd1, sv_embd1), 1)
            embd2 = torch.cat((emo_embd2, sv_embd2), 1)

        if version == 0 or version == 1:
            # Cross-attention
            # Q, K, V
            emo_embd1 = self.emo_projection(emo_embd1)
            emo_embd2 = self.emo_projection(emo_embd2)

            emo_embd1 = emo_embd1.view(emo_embd1.shape[0], 4, self.sv_size // 4)
            sv_embd1 = sv_embd1.view(emo_embd1.shape[0], 4, self.sv_size // 4)
            emo_embd2 = emo_embd2.view(emo_embd1.shape[0], 4, self.sv_size // 4)
            sv_embd2 = sv_embd2.view(emo_embd1.shape[0], 4, self.sv_size // 4)

            embd1, attn_output_weights1 = self.cross_attention(sv_embd1, emo_embd1, emo_embd1)
            embd2, attn_output_weights2 = self.cross_attention(sv_embd2, emo_embd2, emo_embd2)
            embd1 = embd1.reshape(emo_embd1.shape[0], self.sv_size)
            embd2 = embd2.reshape(emo_embd1.shape[0], self.sv_size)

        x1 = self.flow_network(embd1)
        x2 = self.flow_network(embd2)

        return x1, x2
