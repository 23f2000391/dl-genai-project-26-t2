from utils import load_test,load_train
from bilstm import bilstm_predict_test
from rag import rag_predict_test
from tfidf import tfidf_predict_test
from zero_shot import zero_shot_predict_test

def main():
    train=load_train()
    test=load_test()
    test_predictions=bilstm_predict_test(test)
    #test_predictions=zero_shot_predict_test(test)
    #test_predictions=rag_predict_test(test,train)
    #test_predictions=tfidf_predict_test(test,train)
    

    test["prediction"]=test_predictions
    test.to_csv("../data/submission.csv",index=False)
    print("Submission saved.")

if __name__ == "__main__":
    main()