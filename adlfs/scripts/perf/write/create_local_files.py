import time
from io import BytesIO
import os
import torch

from azstoragetorch.io import BlobIO
from transformers import RobertaModel, BertModel
from adlfs import AzureBlobFileSystem
import fsspec
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(30, 30)  # very small layer


def main():
    # content = b"1" * 5 * 1024**3
    model = TinyModel()
    state = model.state_dict()
    with open("5KiB-model.pth", "wb") as f:
        torch.save(state, f)
  
if __name__ == "__main__":
    main()