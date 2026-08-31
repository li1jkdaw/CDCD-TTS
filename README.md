# CDCD-TTS

This repository contains training and inference codes for the text-to-speech model based on Continuous Diffusion for Categorical Data (CDCD-TTS). The codes rely on torch==2.0.1 and torchaudio==2.0.2. To install necessary dependencies, you can run

```
pip install requirements.txt
```

Also, please download *modules* folder by [this link](https://drive.google.com/drive/folders/1nqIFF29RK4cKe7xEbHuLcgIsCFvqwq4Z?usp=sharing) and put it in the main *CDCD-TTS/* directory. CDCD-TTS is the model borrowing many modules from CosyVoice2, and these modules are accessible by the provided link.

## Inference

To run the pre-trained CDCD-TTS model, you can check *codes/inference.ipynb* jupyter notebook. Please download *ckpt* folder with the English checkpoint by [the link](https://drive.google.com/drive/folders/1nqIFF29RK4cKe7xEbHuLcgIsCFvqwq4Z?usp=sharing) and put it in the main *CDCD-TTS/* directory.

This model was trained on English lowercase texts with the limited set of punctuation marks. In the notebook you can see examples of how input texts should look like. Please pay attention that only admissible characters (26 lowercase latin characters, space and puctuation marks) are left during text preprocessing, all other characters are omitted.

Since speech token generation is done non-autoregressively in CDCD-TTS, duration predictor is needed. Simple statistical (non-neural) duration predictor is employed, *inference.ipynb* notebook can be checked to see how it works.

Additionally, you can check *codes/extract-speech-tokens.ipynb* notebook to check how to extract speech tokens. CDCD-TTS operates on CosyVoice2 speech tokens coming from FSQ quantization process.

## Training

## Acknowledgements

CDCD-TTS relies heavily on [CosyVoice2](https://github.com/QwenAudio/CosyVoice). Essentially, CDCD-TTS is CosyVoice2 with LLM speech token generator replaced with the one based on categorical diffusion. The code related to this diffusion is located at *codes/cdcd/* while the remaining folders in *codes/* are borrowed from CosyVoice2 repository and slightly refactored.

## Citation

CDCD-TTS model is described in the paper cited as

```
@inproceedings{cdcd-tts,
               title={Optimality of {FSQ} Tokens for Continuous Diffusion for Categorical Data with Application to Text-to-Speech},
               author={Vadim Popov and Wenju Gu and Tasnima Sadekova and Georgii Aparin and Assel Yermekova},
               booktitle={Forty-third International Conference on Machine Learning},
               year={2026},
               url={https://openreview.net/forum?id=0uS3P0Dlh9}
}
```
