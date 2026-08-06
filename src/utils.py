import wandb

#from chonkie import TokenChunker

import numpy as np              
import pandas as pd             
import matplotlib.pyplot as plt 
import seaborn as sns           
import torch                    
import torch.nn as nn         
from torch.utils.data import Dataset,DataLoader
from collections import Counter 
from string import punctuation  
import warnings                 
import string
import re
import os
import unicodedata

from transformers import pipeline,AutoTokenizer,AutoModel,AutoModelForCausalLM,AutoModelForSequenceClassification,AutoModelForMultipleChoice
from transformers.utils import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import faiss 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from tqdm.auto import tqdm
import uuid
import jsonlines

from datasets import load_dataset

plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')
logging.set_verbosity_error()

print(f"PyTorch Version: {torch.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
np.random.seed(42)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''CONFIG={
    "lr":2e-5,
    "loss":"CrossEntropyLoss",
    "model":"DeBERTa",
    "split":"Stratified-K-Fold",
    "epochs":10,
    "folds_num":5,
    "optimizer":"AdamW",
    "scheduler":"ReduceLROnPlateau",
    "tokenizer":"microsoft/deberta-v3-base",
    "batch_size":16,
    "project_name":"23f2000391-t22026",
    "weight_decay":0.01,
}'''

CONFIG={
    "project_name":"23f2000391-t22026",
    "model":"BiLSTM + Attention Score",
    "tokenizer":"bert-base-uncased",
    "batch_size":16,
    "split":"Stratified-K-Fold",
    "folds_num":5,
    "embedding_dim":300,
    "hidden_dim":256,
    "num_layers":2,
    "dropout":0.3,
    "lr":2e-4,
    "weight_decay":1e-2,
    "epochs":10,
    "optimizer":"AdamW",
    "loss":"CrossEntropyLoss",
    "scheduler":"ReduceLROnPlateau"
}

GLOBAL_choices=["A","B","C","D","E"]

label2id={
    'A':0,
    'B':1,
    'C':2,
    'D':3,
    'E':4
}
id2label={
    0:"A",
    1:"B",
    2:"C",
    3:"D",
    4:"E"
}

def clean_text(text):
    text=str(text)
    text=unicodedata.normalize("NFKC", text)
    text=text.strip()
    text=re.sub(r"\s+", " ", text)
    return text

PREFIXES=[
    "Choose the correct answer:",
    "Pick the best possible answer:",
    "Determine the correct option:",
    "Select the most accurate option:",
    "Identify the correct statement:",
    "Which of the following is correct?"
]

SUFFIXES=[
    "based on the given context.",
    "from the following choices.",
    "among the listed options.",
    "carefully."
]

def clean_prompt(text):
    text=text.strip()

    for p in PREFIXES:
        if text.startswith(p):
            text=text[len(p):].strip()

    for s in SUFFIXES:
        if text.endswith(s):
            text=text[:-len(s)].strip()

    return text

def load_train():
    train=pd.read_csv("data/train.csv")

    train.drop(columns="id", inplace=True)

    train["correct_option"]=train.apply(
        lambda row: row[row["answer"]],
        axis=1
    )

    for col in ["prompt", "A", "B", "C", "D", "E"]:
        train[f"clean_{col}"]=train[col].apply(clean_text)

    train["clean_prompt"]=train["clean_prompt"].apply(clean_prompt)

    train["combined_text"]=(
        "Question: " + train["clean_prompt"] +
        "\n\nA: " + train["A"] +
        "\nB: " + train["B"] +
        "\nC: " + train["C"] +
        "\nD: " + train["D"] +
        "\nE: " + train["E"]
    )

    return train

def load_test():
    test=pd.read_csv("data/test.csv")
    test.drop(columns="id", inplace=True)

    for col in ["prompt", "A", "B", "C", "D", "E"]:
        test[f"clean_{col}"]=test[col].apply(clean_text)

    test["clean_prompt"]=test["clean_prompt"].apply(clean_prompt)

    test["combined_text"]=(
        "Question: " + test["clean_prompt"] +
        "\n\nA: " + test["A"] +
        "\nB: " + test["B"] +
        "\nC: " + test["C"] +
        "\nD: " + test["D"] +
        "\nE: " + test["E"]
    )

    return test