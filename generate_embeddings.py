import os
import numpy as np
from PIL import Image
import shutil
import pandas as pd
import torchaudio
import torch
from huggingface_hub import hf_hub_download
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, AutoModel
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import sys


target_sample_rate = 16000

root_dir = ''
classes = sorted(os.listdir(root_dir))

output_dir = ''
os.makedirs(output_dir, exist_ok=True)


# Define model
model_arr = ["ecapa2", "ecapa", "resnet", "wav2vec", "emotion2vec"]
model = model[0]

if model == "ecapa2":
	model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)
	encoder =  torch.jit.load(model_file, map_location='cuda')

if model == "ecapa":
	encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})

if model == "resnet":
	encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-resnet-voxceleb", run_opts={"device":"cuda"})

if model == "wav2vec":
	encoder = AutoModel.from_pretrained("harshit345/xlsr-wav2vec-speech-emotion-recognition")
	processor = Wav2Vec2FeatureExtractor.from_pretrained("harshit345/xlsr-wav2vec-speech-emotion-recognition")
	encoder.eval()
	encoder.to('cuda')


if model == "emotion2vec":
	encoder = pipeline(task=Tasks.emotion_recognition, model="iic/emotion2vec_base", device="cuda")


class suppress_output:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# List files as an array
for class_name in classes:
    print(class_name)
    if class_name == "audio_speech_actors_01-24": 
        continue
    class_dir = os.path.join(root_dir, class_name)
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
            
        class_dir_output = os.path.join(output_dir, class_name)
        os.makedirs(class_dir_output, exist_ok=True)
            
        waveform, sample_rate = torchaudio.load(file_path)

            
        if sample_rate != target_sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0).unsqueeze(0) 

        waveform = waveform.to('cuda')
        # Extract embeddings

		if model == "ecapa2":
			with torch.jit.optimized_execution(False):
				embd = encoder(x = waveform,  labels = "embedding").squeeze(dim=0)
				
		if model == "ecapa" or model == "resnet":
			embd = encoder.encode_batch(waveform).squeeze(dim=1).squeeze(dim=0)
			
			
		if model == "wav2vec":
			inputs_processed_emo = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
			inputs_processed_emo.input_values = inputs_processed_emo.input_values.squeeze(dim=0).to('cuda')
			embd = encoder(inputs_processed_emo.input_values) # seq x 1024
			embd = embd.last_hidden_state
			embd = embd.mean(dim=1).squeeze(dim=0)

		if model == "emotion2vec":
			with suppress_output():
				inputs_processed = torch.clone(waveform)
				inputs_processed = list(inputs_processed)
				outputs = encoder(input = inputs_processed, granularity="utterance", extract_embedding=True)
			embd = []
			for output in outputs:
				embd.append(torch.tensor(output['feats']))
			embd = torch.stack(embd).squeeze(dim=0)
        
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        np.save(os.path.join(class_dir_output, f"{file_name}.npy"), embd.detach().cpu().numpy())
        # Save embeddings as [embd_dim]
