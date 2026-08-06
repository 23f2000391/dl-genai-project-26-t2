from utils import *

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

