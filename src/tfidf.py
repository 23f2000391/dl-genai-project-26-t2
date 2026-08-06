from utils import *
from metrics import *
def build_pairwise_dataset(df,test=False):
    rows=[]
    options=GLOBAL_choices
    for qid,row in df.iterrows():
        for c in options:
            if not test:
                rows.append({
                    "text":"Question: "+row["clean_prompt"]+"\nAnswer: "+row[c],
                    "label":1 if c==row["answer"] else 0,
                    "question_id":qid,
                    "option":c
                })
            else:
                rows.append({
                    "text":"Question: "+row["clean_prompt"]+"\nAnswer: "+row[c],
                    "question_id":qid,
                    "option":c
                })

    return pd.DataFrame(rows)

def load_models():
    vectorizer=TfidfVectorizer(
        ngram_range=(1,2),
        stop_words="english",
        min_df=3,
        sublinear_tf=True
    )
    clf=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42)
    return vectorizer,clf

def train_tfidf(train,skf):
    fold_map3=[]
    fold_acc=[]
    fold_top3=[]
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train["answer"])):
        vectorizer,clf=load_models()
        train_df=train.iloc[train_idx].reset_index(drop=True)
        val_df=train.iloc[val_idx].reset_index(drop=True)
        
        print("Fold:",fold+1)
        
        pair_train=build_pairwise_dataset(train_df)
        pair_val=build_pairwise_dataset(val_df)

        X_train=vectorizer.fit_transform(pair_train["text"])
        X_val=vectorizer.transform(pair_val["text"])

        clf.fit(X_train,pair_train["label"])
        
        probs=clf.predict_proba(X_val)[:,1]

        predictions=[]
        choices=GLOBAL_choices
        for i in range(len(val_df)):
            start=i*5
            end=start+5
            scores=probs[start:end]
            ranked=np.argsort(scores)[::-1]
            top3=[choices[j] for j in ranked[:3]]
            predictions.append(top3)
        
        map3=map_at_3(val_df["answer"],predictions)
        acc=top1_accuracy(val_df["answer"],predictions)
        top3=top3_accuracy(val_df["answer"],predictions)

        print(f"MAP@3: {map3:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Top3 Accuracy: {top3:.4f}")

        fold_map3.append(map3)
        fold_acc.append(acc)
        fold_top3.append(top3)

        '''wandb.log({
            "Fold": fold + 1,
            "MAP@3": map3,
            "Top1 Accuracy": acc,
            "Top3 Accuracy": top3
        })'''
    print("Average MAP@3:", np.mean(fold_map3))
    print("Average Accuracy:", np.mean(fold_acc))
    print("Average Top3 Accuracy:", np.mean(fold_top3))

def tfidf_predict_test(test,train):
    vectorizer,clf=load_models()
    pair_train=build_pairwise_dataset(train)
    X_train=vectorizer.fit_transform(pair_train["text"])
    clf.fit(X_train,pair_train["label"])

    pair_test=build_pairwise_dataset(test,True)
    X_test=vectorizer.transform(pair_test["text"])
    probs=clf.predict_proba(X_test)[:, 1]

    test_predictions=[]
    for i in range(len(test)):
        scores=probs[i*5:(i+1)*5]
        ranked=np.argsort(scores)[::-1]
        top3=[GLOBAL_choices[j] for j in ranked[:3]]
        test_predictions.append(" ".join(top3))

    return test_predictions