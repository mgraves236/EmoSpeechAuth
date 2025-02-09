dataset_dir = './datasets/all-'

# Choose upstream model (frontend) for emotion embeddings
emo_model_array = ["emotion2vec", "wav2vec"]
emo_model_name = emo_model_array[0]

# Choose upstream model (frontend) for speaker verification embeddings
sv_model_array = ["ecapa-tdnn", "ecapa2", "resnet"]
sv_model_name = sv_model_array[0]

# Training hyperparameters
batch_size = 8
num_epochs = 100
learning_rate = 1e-4
save_interval = 5
dropout_v = 0.4

seeds = [93829758, 93847482, 69620752, 85127261, 51385573]
rd_seed = seeds[0]

# Choose version of the model
# 0 -  cross-attention
# 1 - cross-attention with mish
# 2 - concatenation
version = 0

