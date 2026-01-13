# EmoSpeechAuth: Emotion-Aware Speaker Verification 

This repository contains the code used in the paper:

> 📄 **EmoSpeechAuth: Emotion-Aware Speaker Verification**  
> Magdalena Gołębiowska, Piotr Syga,
> Wrocław University of Science and Technology, 
> *Interspeech, 2025*

Access the paper on the [ISCA archive](https://www.isca-archive.org/interspeech_2025/goebiowska25_interspeech.html#).
## 🧠 Overview

 We propose a novel framework for constructing emotional speaker embeddings. Our framework utilizes pretrained state-of-the-art feature extractors for speaker and emotion recognition, including both speaker and emotional information in the final embeddings.

We provide:

- Preprocessing scripts to generate fundamental model embedddings,
- Model implementation in PyTorch,
- Training and evaluation routines.

Datasets used in the paper are public and may be accessed for example on Kaggle.

Below is a description of file contents.


| **File name** | **Contents** |
| ------------- | ------------- |
| augmentation.py  | Includes classes that define augmentation on embeddings. |
| config.py  | Choose emotion and speaker upstream model, training hyperparameters, seed, and architecture version. |
| dataset.py  | Defines embeddings dataset class and pair creation for contrastive learning. |
| e2_e2v_seed0.pth | Model checkpoint for the best architecture variant with ECAPA2 and emtotion2vec.  |
| generate_embeddings.py | Code to generate frontend embeddings which create the final dataset.|
| model.py | Defines model architecutre and contrastive loss.  |
| train.py| Script to train, validate and test the model. |
| utils.py | Auxiliary functions. |

## 📖 Example Usage

Minimal working example to verify one raw audio sample against another one.

First you need to extract embeddings using [generate_embeddings.py](https://github.com/mgraves236/EmoSpeechAuth/blob/main/generate_embeddings.py) for both speaker and emotional encoder. 

The script expects the directory with structure defined as:
```
    root_dataset
    |__ speaker_1
    |__ speaker_2
    |__ speaker_3
    ...
```
To define what model to use to extract embeddings, use this line:
```Python
# Define model
model_arr = ["ecapa2", "ecapa", "resnet", "wav2vec", "emotion2vec"]
model = model_arr[0]
```
Running the script will create a directory as specified by ```output_dir = ''``` with ```.npy``` files.

Then you need to run [extract_embeddings.py](https://github.com/mgraves236/EmoSpeechAuth/blob/main/extract_embeddings.py) which will generate EmoSpeechAuth embeddings. 
Loading the model from the provided checkpoint is included in the script.
Make sure to provide your dataset path with ```.npy``` files in the extract_embeddings.py file:
```Python
root_dir = '' + sv_model_name
classes = sorted(os.listdir(root_dir))
```

## 📝 Citing

If you use our work in your research, please cite:
```
@inproceedings{goebiowska25_interspeech,
  title     = {{EmoSpeechAuth: Emotion-Aware Speaker Verification}},
  author    = {{Magdalena Gołębiowska and Piotr Syga}},
  year      = {{2025}},
  booktitle = {{Interspeech 2025}},
  pages     = {{5743--5747}},
  doi       = {{10.21437/Interspeech.2025-515}},
  issn      = {{2958-1796}},
}

```
