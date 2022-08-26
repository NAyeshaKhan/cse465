import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from spacy.lang.en import English
spacy_eng = English()

# create vocabulary to map each word to an index
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
	@@ -24,21 +34,21 @@ def tokenizer_eng(text):

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        index = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = index
                    self.itos[index] = word
                    index += 1

    # convert text to numerical values
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

	@@ -47,18 +57,18 @@ def numericalize(self, text):
            for token in tokenized_text
        ]

# setup pytorch dataset to load the data
class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get img, captions column
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize and build vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

	@@ -80,59 +90,53 @@ def __getitem__(self, index):
        return img, torch.tensor(numericalized_caption)


# padding the images to be of the same length
class MyCollate:
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_index)

        return images, targets


def get_loader(
        root_folder,
        annotation_file,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,

):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    pad_index = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_index=pad_index),
    )
    return loader


def main():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataloader = get_loader("flickr8K/images/",
                            annotation_file="flickr8k/captions.txt",
                            transform=transform)
    for index, (images, captions) in enumerate(dataloader):
        print(images.shape)
        print(captions.shape)


if __name__ == "__main__":
    main()
