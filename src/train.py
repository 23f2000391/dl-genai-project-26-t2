from utils import load_train,StratifiedKFold
from rag import train_rag
from bilstm import train_bilstm
from tfidf import train_tfidf
from zero_shot import train_zero_shot

def main():
    train=load_train()
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    train_bilstm(train,skf)
    #train_zero_shot(train,skf)
    #train_tfidf(train,skf)
    #train_rag(train,skf)

if __name__=="__main__":
    main()