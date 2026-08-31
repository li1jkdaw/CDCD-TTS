# CDCD-TTS

This repository contains training and inference codes for the text-to-speech model based on Continuous Diffusion for Categorical Data (CDCD-TTS) described in [this paper](https://arxiv.org/abs/2606.09962). The codes rely on torch==2.0.1 and torchaudio==2.0.2. To install necessary dependencies, you can run

```
pip install -r requirements.txt
```

Also, please download *modules/* folder by [this link](https://drive.google.com/drive/folders/1nqIFF29RK4cKe7xEbHuLcgIsCFvqwq4Z?usp=sharing) and put it in the main *CDCD-TTS/* directory. CDCD-TTS is the model borrowing many modules from CosyVoice2, and the provided link gives access to the corresponding checkpoints.

## Inference

To run the pre-trained CDCD-TTS model, you can check *codes/inference.ipynb* jupyter notebook. Please download *ckpt/* folder with the English checkpoint by [the link](https://drive.google.com/drive/folders/1nqIFF29RK4cKe7xEbHuLcgIsCFvqwq4Z?usp=sharing) and put it in the main *CDCD-TTS/* directory.

This model was trained on English lowercase texts with the limited set of punctuation marks. In the notebook you can see examples of how input texts should look like. Please pay attention that only admissible characters (26 lowercase latin characters, space and puctuation marks) are left during text preprocessing, all other characters are omitted.

Since speech token generation is done non-autoregressively in CDCD-TTS, duration predictor is needed. Simple statistical (non-neural) duration predictor is employed, *inference.ipynb* notebook can be checked to see how it works.

Additionally, you can check *codes/extract-speech-tokens.ipynb* notebook to check how to extract speech tokens for a single audio. CDCD-TTS operates on CosyVoice2 speech tokens coming from FSQ quantization process.

## Training

To train your model, please run something like this:

```
cd codes/cdcd && accelerate launch train.py --data_folder /path/to/data/ --dataset_names "dataset1,dataset2,dataset3" --n_epochs 10 --checkpoint_folder /path/to/checkpoints/folder/
```

In this example there are three different datasets located at */path/to/data/dataset1*, */path/to/data/dataset2*, */path/to/data/dataset3*, each folder having multiple triplets of files, e.g. (data_0_info.txt, data_0_speech.npz, data_0_text.npz), (data_1_info.txt, data_1_speech.npz, data_1_text.npz), etc.

For each triplet:
1. *_info.txt contains lines "duration in seconds"|"any other unused info like some general id of the utterance in the dataset" for each spoken utterance
1. *_speech.npz contains numpy arrays with FSQ speech token ids of dtype=np.int16 (see *codes/extract-speech-tokens.ipynb*) for the corresponding utterances
1. *_text.npz contains numpy arrays with text character ids of dtype=np.int16 (see mapping in *codes/dataset.py*) for the corresponding utterances

In our experiments, for each dataset we arranged data into several npz files of reasonable size (check *data/* folder) due to some upload/download/read issues.

For your reference, the pre-trained model in *ckpt/* folder [here](https://drive.google.com/drive/folders/1nqIFF29RK4cKe7xEbHuLcgIsCFvqwq4Z?usp=sharing) was trained on a single node with 8 V100. The folder also contains the training curve with the values of diffusion loss.

## Acknowledgements

CDCD-TTS relies heavily on [CosyVoice2](https://github.com/QwenAudio/CosyVoice). Essentially, CDCD-TTS is CosyVoice2 with LLM speech token generator replaced with the one based on categorical diffusion. The code related to this diffusion is located at *codes/cdcd/* while the remaining folders in *codes/* are borrowed from CosyVoice2 repository and slightly refactored. The diffusion backbone borrows much of its architecture from [F5-TTS](https://github.com/swivid/f5-tts).

## Citation

CDCD-TTS model is described in the paper accepted at ICML. Please cite it as

```
@inproceedings{cdcd-tts,
               title={Optimality of {FSQ} Tokens for Continuous Diffusion for Categorical Data with Application to Text-to-Speech},
               author={Vadim Popov and Wenju Gu and Tasnima Sadekova and Georgii Aparin and Assel Yermekova},
               booktitle={Forty-third International Conference on Machine Learning},
               year={2026},
               url={https://openreview.net/forum?id=0uS3P0Dlh9}
}
```
