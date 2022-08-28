## CSE465 Project Summer 2022

## Project: Image Captioning Using CNN 

#### Requirements: `Pytorch, Torch Vision, Spacy, Tensorboard, PIL, TQDM`

python -m spacy download en
Download the dataset: [link](https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb) 
and put it under `./data` make sure you unzip file folder and rename it from `flicker8k` to `data`

- Set images folder, captions.txt inside a folder Flickr8k.
- train.py: For training the CNN model.
- model.py: creating the encoderCNN, decoderRNN model and combine them together into one model. 
- get_loader.py: Loading the data, creating vocabulary for training, testing and evluation.
- utils.py: Load model, save model, printing few test cases to evaluate the model.
- test.py: for testing the model with some example test images


#### Usage: For Training Start running the training script with `python train.py` 
#### Usage: Use test.ipynb notebook to test the model and play with it.

