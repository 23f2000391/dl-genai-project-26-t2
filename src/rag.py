from utils import *
from metrics import *
embedder=None
cross_encoder=None

def get_embedder():
    global embedder
    if embedder is None:
        embedder=SentenceTransformer("BAAI/bge-base-en-v1.5") 
    return embedder

def get_cross_encoder():
    global cross_encoder
    if cross_encoder is None:
        cross_encoder=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return cross_encoder

def build_kb(df):   
    kb=[]
    embedder=get_embedder()
    for idx, row in df.iterrows(): 
        kb.append({
            "prompt": row["clean_prompt"],
            "combined": row["combined_text"],
            "A": row["A"],
            "B": row["B"],
            "C": row["C"],
            "D": row["D"],
            "E": row["E"],
            "answer": row["answer"]
        })
    prompt_texts=[i["combined"] for i in kb]

    kb_embeddings=embedder.encode(
        prompt_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    
    dim=kb_embeddings.shape[1]
    index=faiss.IndexFlatIP(dim)
    index.add(kb_embeddings)
    return kb,index

def retrieve(query,kb,index,k=50):
    embedder=get_embedder()
    emb=embedder.encode(query,convert_to_numpy=True,normalize_embeddings=True).reshape(1,-1)

    scores,indices=index.search(emb,k)

    retrieved=[]
    for score,idx in zip(scores[0],indices[0]):
        item=kb[idx].copy()
        item["faiss_score"]=float(score)
        retrieved.append(item)
    return retrieved

def rerank(query,retrieved):
    cross_encoder=get_cross_encoder()
    pairs=[[query,item["prompt"]] for item in retrieved]
    ce_scores=cross_encoder.predict(pairs)

    for i,s in enumerate(ce_scores):
        retrieved[i]["ce_score"]=float(s)
    retrieved=sorted(retrieved,key=lambda x:x["ce_score"],reverse=True)
    return retrieved

def rag_predict(row,kb,index):
    query=(
        "Question: "+row["clean_prompt"] +
        "\n\nA: "+row["A"] +
        "\nB: "+row["B"] +
        "\nC: "+row["C"] +
        "\nD: "+row["D"] +
        "\nE: "+row["E"]
    )
    retrieved=retrieve(query,kb,index,k=50)
    retrieved=rerank(query,retrieved)
    top3=[]
    for i in retrieved:
        if i["answer"] not in top3:
            top3.append(i["answer"])
        if len(top3)==3:
            break
    
    return top3

def train_rag(train,skf):
    fold_map3=[]
    fold_acc=[]
    fold_top3=[]
    fold_f1=[]
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train["answer"])):
        train_df=train.iloc[train_idx].reset_index(drop=True)
        val_df=train.iloc[val_idx].reset_index(drop=True)
        
        print("Fold:",fold+1)
        
        kb,index=build_kb(train_df)

        predictions=[]
        for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
            top3_preds=rag_predict(row,kb,index)        
            predictions.append(top3_preds)
        
        map3=map_at_3(val_df["answer"],predictions)
        acc=top1_accuracy(val_df["answer"],predictions)
        top3=top3_accuracy(val_df["answer"],predictions)
        f1=macro_f1_score(val_df["answer"],predictions)
        
        print(f"MAP@3:{map3:.4f}")
        print(f"Accuracy:{acc:.4f}")
        print(f"Top3 Accuracy:{top3:.4f}")
        print(f"Macro F1:{f1:.4f}")
        
        fold_map3.append(map3)
        fold_acc.append(acc)
        fold_top3.append(top3)
        fold_f1.append(f1)
        '''wandb.log({
            "Fold":fold+1,
            "MAP@3":map3,
            "Top1 Accuracy":acc,
            "Top3 Accuracy":top3,
            "Macro F1 Score":f1
        })'''

    print("Average MAP@3:", np.mean(fold_map3))
    print("Average Accuracy:", np.mean(fold_acc))
    print("Average Top3 Accuracy:", np.mean(fold_top3))
    print("Average Macro F1 Score:", np.mean(fold_f1))

    '''wandb.log({
        "Average MAP@3": np.mean(fold_map3),
        "Average Top1 Accuracy": np.mean(fold_acc),
        "Average Top3 Accuracy": np.mean(fold_top3),
        "Average Macro F1 Score": np.mean(fold_f1)
    })'''

def rag_predict_test(test,train):
    group_train=train.groupby(GLOBAL_choices+['clean_prompt'])[['answer','combined_text']].first().reset_index()
    test_predictions=[]
    kb_train,index_train=build_kb(group_train)
    for _, row in tqdm(test.iterrows(), total=len(test)):
        test_predictions.append(rag_predict(row,kb_train,index_train))
    return test_predictions