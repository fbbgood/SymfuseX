import os
import random
import dgl
import logging
import torch
import numpy as np

CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9,
    "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17,
    "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
}
CHARPROTLEN = 25

def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def graph_collate_func(x):

    if len(x[0]) == 4:
        d, p, y, smiles = zip(*x)
        d = dgl.batch(d)
        return d, torch.tensor(np.array(p)), torch.tensor(y), list(smiles)
    else:
        d, p, y = zip(*x)
        d = dgl.batch(d)
        return d, torch.tensor(np.array(p)), torch.tensor(y)

def mkdir(path):
    path = path.strip().rstrip("\\")
    if not os.path.exists(path):
        os.makedirs(path)

def integer_label_protein(sequence, max_length=1200):
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            encoding[idx] = CHARPROTSET[letter.upper()]
        except KeyError:
            logging.warning(f"character {letter} does not exists in sequence category encoding, treat as padding.")
    return encoding
