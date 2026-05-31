import torch
from tokenizers.pre_tokenizers import BertPreTokenizer, Whitespace
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class TextDataset(Dataset):
    def __init__(self, text: str, context_window=256):
        self.stride = context_window
        self.text = text.replace("\n", " \n ")
        self.context_window = context_window
        self.pre_tokenizer = Whitespace()
        try:
            self.tokenizer = Tokenizer.from_file("data/tokenizer.json")
            self.tokenizer.pre_tokenizer = self.pre_tokenizer
        except:
            self.tokenizer = Tokenizer(BPE())
            self.tokenizer.pre_tokenizer = self.pre_tokenizer
            trainer = BpeTrainer(
                vocab_size=10000,
                special_tokens=[
                    "<PAD>",
                    "<BOS>",
                    "<EOS>",
                    "<UNK>"
                ]
            )
            self.tokenizer.train_from_iterator([self.text], trainer)
            self.tokenizer.save("data/tokenizer.json")

        self.tokenized_text = torch.tensor(self.tokenizer.encode(self.text).ids, dtype=torch.long)

    # def __len__(self):
    #     return len(self.tokenized_text) - self.context_window
    #
    # def __getitem__(self, i):
    #     c = self.context_window
    #     return self.tokenized_text[i:i+c], self.tokenized_text[i+1:i+c+1]

    def __len__(self):
        return (len(self.tokenized_text) - self.context_window) // self.stride

    def __getitem__(self, i):
        s = i * self.stride
        c = self.context_window
        return self.tokenized_text[s:s + c], self.tokenized_text[s + 1:s + c + 1]

