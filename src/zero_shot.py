from utils import *
from metrics import *
def get_qwen():
    if qwen_tokenizer is None:
        qwen_model_name="Qwen/Qwen2.5-7B-Instruct"
        qwen_tokenizer=AutoTokenizer.from_pretrained(qwen_model_name)
        qwen_model=AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            dtype=torch.float16,
            device_map="auto"
        )
    return qwen_tokenizer,qwen_model

def create_zero_shot_prompt(row):
    zero_shot_prompt=f"""
    You are solving a multiple choice question containing 5 choices.
    Question:
    {row["clean_prompt"]}
    Choices:
    A. {row["A"]}
    B. {row["B"]}
    C. {row["C"]}
    D. {row["D"]}
    E. {row["E"]}
    Return ONLY the three most likely answer labels with a single space separating them.
    Example:
    C A D
    """
    return zero_shot_prompt

def create_few_shot_prompt(row,df):
    few_shot_examples = []
    for label in GLOBAL_choices:
        example = df[df["answer"] == label].sample(n=1,random_state=42).iloc[0]
        few_shot_examples.append(example)
    prompt = """You are an expert at solving multiple-choice questions.
                Below are some solved examples.\n"""

    for i, ex in enumerate(few_shot_examples, 1):

        prompt += f"""Example {i}
        Question: {ex["clean_prompt"]}
        
        Choices:
        A. {ex["A"]}
        B. {ex["B"]}
        C. {ex["C"]}
        D. {ex["D"]}
        E. {ex["E"]}
        
        Correct Answer:
        {ex["answer"]}
        
        """
        
    prompt += f"""
    Now answer the following question.
    
    Question:
    {row["clean_prompt"]}
    
    Choices:
    A. {row["A"]}
    B. {row["B"]}
    C. {row["C"]}
    D. {row["D"]}
    E. {row["E"]}
    
    Rank the choices based on their probability of being the correct answer.
    Then return the top three answer labels separated by spaces.
    
    Example output:
    C A D
    
    Do not explain your answer.
    """

    return prompt

def predict_qwen(row):
    qwen_tokenizer,qwen_model=get_qwen()
    prompt = create_few_shot_prompt(row)

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = qwen_tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    ).to(qwen_model.device)

    with torch.no_grad():

        outputs = qwen_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=qwen_tokenizer.eos_token_id
        )

    generated = outputs[0][inputs.input_ids.shape[1]:]

    prediction = qwen_tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    labels = re.findall(r"\b[A-E]\b", prediction)

    return labels[:3]

def train_zero_shot(train,skf):

    fold_map=[]
    fold_acc=[]
    fold_top3=[]
    fold_f1=[]
    for fold,(train_idx, val_idx) in enumerate(skf.split(train, train["answer"])):
        train_df=train.iloc[train_idx].reset_index(drop=True)
        val_df=train.iloc[val_idx].reset_index(drop=True)
        predictions=[]
        for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
            pred=predict_qwen(row)
            predictions.append(pred)
            torch.cuda.empty_cache()
        map3=map_at_3(
            val_df["answer"],
            predictions
        )
        
        acc=top1_accuracy(
            val_df["answer"],
            predictions
        )
        
        top3=top3_accuracy(
            val_df["answer"],
            predictions
        )
        f1=macro_f1_score(val_df["answer"],predictions)
        '''wandb.log({
            "Fold": fold + 1,
            "MAP@3": map3,
            "Accuracy": acc,
            "Macro F1":f1,
            "Top3 Accuracy": top3
        })'''
        fold_map.append(map3)
        fold_acc.append(acc)
        fold_top3.append(top3)
        fold_f1.append(f1)
    print(f"MAP@3 : {np.mean(fold_map):.4f}")
    print(f"Accuracy : {np.mean(fold_acc):.4f}")
    print(f"Top3 Accuracy : {np.mean(fold_top3):.4f}")
    print(f"Macro F1 Score : {np.mean(fold_f1):.4f}")

def zero_shot_predict_test(test):
    test_predictions=[]
    for _, row in tqdm(test.iterrows(), total=len(test)):
        test_predictions.append(
            predict_qwen(row)
        )
    return test_predictions