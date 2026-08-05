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

CONFIG={
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
}

GLOBAL_choices=["A","B","C","D","E"]

label2id={
    'A':0,
    'B':1,
    'C':2,
    'D':3,
    'E':4
}
id2label = {
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

def map_at_3(true,pred):
    scores=[]
    for actual,preds in zip(true,pred):
        score=0.0
        for rank,pred in enumerate(preds,start=1):
            if pred==actual:
                score=1.0/rank
                break
        scores.append(score)
    return np.mean(scores)

def top3_accuracy(y_true, predictions):
    return np.mean([truth in pred for truth, pred in zip(y_true, predictions)])

def top1_accuracy(y_true, predictions):
    top1=[pred[0] if len(pred) else "Z" for pred in predictions]

    return accuracy_score(y_true,top1)

def macro_f1_score(y_true, predictions):
    top1_preds=[]

    for pred in predictions:
        top1_preds.append(pred[0])

    return f1_score(y_true,top1_preds,average="macro",labels=["A", "B", "C", "D", "E"],zero_division=0)

