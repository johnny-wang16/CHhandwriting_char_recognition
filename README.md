# Traditional chinese character recognition

This repo contains training code to make a traditional chinese character classifier that recognize 13065 chinese characters using the open-source [Traditional Chinese Handwriting Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset.git). It's using Resnet18 architechture and it currently achieves 98.92% accuracy on the test set.
The model can be downloaded from this [google drive link](https://drive.google.com/file/d/1ngzmc3De8MGS8pmOO0XPV1LwjrMry98E/view?usp=sharing)

## File explanation:

train.py: main file that contains DNN code

document2chars.ipynb: main file that contains code to process document images to extract useful information.


## Usage:
- To train the model from scratch, run: `python train.py --train`
- To evaluate the model on the validation set, run: `python train.py --evaluate`
- To evaluate a single image, modify the code to read in the image and run: `python train.py --single-evaluate`

