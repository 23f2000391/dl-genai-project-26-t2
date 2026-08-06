from utils import *
from metrics import *

tokenizer=None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer=AutoTokenizer.from_pretrained(CONFIG["tokenizer"])
    return tokenizer


class Attention(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.linear=nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self,x):
        weights=torch.softmax(self.linear(x),dim=1)
        context=(weights*x).sum(dim=1)
        return context

class BiLSTMAttention(nn.Module):
    def __init__(self,vocab_size,embedding_dim=300,hidden_dim=256,num_layers=2,dropout=0.3):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,num_layers=num_layers,batch_first=True,bidirectional=True,dropout=dropout)
        self.attention=Attention(hidden_dim)
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(hidden_dim*2,1)
        self.layernorm=nn.LayerNorm(hidden_dim * 2)
        self.classifier=nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self,input_ids):
        scores=[]
        for i in range(5):
            x=input_ids[:,i]
            x=self.embedding(x)
            output,_=self.lstm(x)
            context=self.attention(output)
            context=self.layernorm(context)
            context=self.dropout(context)
            score=self.classifier(context)
            scores.append(score.squeeze(1))
        scores=torch.stack(scores,dim=1)
        return scores

def encode_question(question,option):
    tokenizer=get_tokenizer()
    encoding=tokenizer(
        question,
        option,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_attention_mask=False,
        return_token_type_ids=False
    )

    return encoding["input_ids"]

class MCQDataset(Dataset):
    def __init__(self, dataframe):
        self.df=dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        inputs=[]
        for choice in GLOBAL_choices:
            ids=encode_question(row["clean_prompt"],row[f"clean_{choice}"])
            inputs.append(ids)

        inputs=torch.tensor(inputs)
        label=label2id[row["answer"]]
        return {"input_ids": inputs,"label": torch.tensor(label)}

def train_one_epoch(model,dataloader,optimizer,criterion,device):
    model.train()
    running_loss=0

    for batch in tqdm(dataloader):
        input_ids=batch["input_ids"].to(device)
        labels=batch["label"].to(device)
        optimizer.zero_grad()
        outputs=model(input_ids)
        loss=criterion(outputs,labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)

        optimizer.step()
        running_loss+=loss.item()

    epoch_loss=running_loss/len(dataloader)

    return epoch_loss

def validate(model,dataloader,criterion,device):
    model.eval()
    running_loss=0
    test_predictions=[]
    ground_truth=[]
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids=batch["input_ids"].to(device)
            labels=batch["label"].to(device)

            outputs=model(input_ids)
            loss=criterion(outputs, labels)
            running_loss+=loss.item()

            probs=torch.softmax(outputs, dim=1)

            top3=torch.topk(probs,k=3,dim=1).indices.cpu().numpy()

            for pred in top3:
                test_predictions.append([id2label[i] for i in pred])

            ground_truth.extend([id2label[i.item()] for i in labels])

    val_loss=running_loss/len(dataloader)
    map3=map_at_3(ground_truth,test_predictions)
    acc=top1_accuracy(ground_truth,test_predictions)
    top3_acc=top3_accuracy(ground_truth,test_predictions)
    f1=macro_f1_score(ground_truth,test_predictions)
    return (val_loss,map3,acc,top3_acc,f1,test_predictions)

def train_bilstm(train,skf):
    fold_map3=[]
    fold_top1=[]
    fold_top3=[]
    fold_f1=[]
    best_cv_map3=-1
    for fold,(train_idx,val_idx) in enumerate(skf.split(train,train["answer"])):
        
        print(f"{'='*10}Fold {fold+1}{'='*10}")

        train_df=train.iloc[train_idx].reset_index(drop=True)
        val_df=train.iloc[val_idx].reset_index(drop=True)

        train_dataset=MCQDataset(train_df)
        val_dataset=MCQDataset(val_df)
        
        train_loader=DataLoader(train_dataset,batch_size=CONFIG["batch_size"],shuffle=True)
        val_loader=DataLoader(val_dataset,batch_size=CONFIG["batch_size"],shuffle=False)

        bilstm_model=BiLSTMAttention(
            vocab_size=get_tokenizer().vocab_size,
            embedding_dim=CONFIG["embedding_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            num_layers=CONFIG["num_layers"],
            dropout=CONFIG["dropout"]
        )
        bilstm_model.to(device)
        
        criterion=nn.CrossEntropyLoss()
        optimizer=torch.optim.AdamW(bilstm_model.parameters(),lr=CONFIG["lr"],weight_decay=CONFIG["weight_decay"])

        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=2)
        
        best_map3=-1
        best_top3=0
        best_top1=0
        best_f1=0
        patience=3
        counter=0

        for epoch in range(CONFIG["epochs"]):

            print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
            train_loss=train_one_epoch(bilstm_model,train_loader,optimizer,criterion,device)
            (val_loss,map3,acc,top3_acc,f1,test_predictions)=validate(bilstm_model,val_loader,criterion,device)
        
            print(f"Train Loss:{train_loss:.4f}")
            print(f"Val Loss  :{val_loss:.4f}")
            print(f"MAP@3     :{map3:.4f}")
            print(f"Top1 Acc  :{acc:.4f}")
            print(f"Top3 Acc  :{top3_acc:.4f}")
            print(f"Macro F1  :{f1:.4f}")

            scheduler.step(map3)

            '''wandb.log({
                "Fold":fold+1,
                "Epoch":epoch+1,
                "Train Loss":train_loss,
                "Validation Loss":val_loss,
                "Validation MAP@3":map3,
                "Validation Top1 Accuracy":acc,
                "Validation Top3 Accuracy":top3_acc,
                "Validation Macro F1":f1,
                "Learning Rate":optimizer.param_groups[0]["lr"]
                })'''

            if map3>best_map3:
                best_map3=map3
                best_top1=acc
                best_top3=top3_acc
                best_f1=f1
                counter=0

                if best_map3>best_cv_map3:
                    best_cv_map3=map3
                    os.makedirs("models",exist_ok=True)
                    torch.save(bilstm_model.state_dict(),"models/bilstm_best_model.pt")
            else:
                counter+=1
                if counter>=patience:
                    print("Early Stopping")
                    break
        fold_map3.append(best_map3)
        fold_top1.append(best_top1)
        fold_top3.append(best_top3)
        fold_f1.append(best_f1)
        '''wandb.log({
            "Fold":fold+1,
            "Fold MAP@3":best_map3,
            "Fold Top1":best_top1,
            "Fold Top3":best_top3,
            "Fold F1":best_f1
            })'''
    print(f"MAP@3: {np.mean(fold_map3):.4f}")
    print(f"Top1 : {np.mean(fold_top1):.4f}")
    print(f"Top3 : {np.mean(fold_top3):.4f}")
    print(f"F1   : {np.mean(fold_f1):.4f}")

    '''artifact=wandb.Artifact("bilstm_best_model",type="model")
    artifact.add_file("/kaggle/working/bilstm_best_model.pt")
    wandb.log_artifact(artifact)
    wandb.finish()'''

class TestDataset(Dataset):
    def __init__(self, dataframe):
        self.df=dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        inputs=[]
        for choice in GLOBAL_choices:
            ids=encode_question(row["clean_prompt"],row[f"clean_{choice}"])
            inputs.append(ids)

        return {"input_ids": torch.tensor(inputs)}

def bilstm_predict_test(test):
    test_dataset=TestDataset(test)
    test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

    bilstm_model = BiLSTMAttention(
        vocab_size=get_tokenizer().vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    ).to(device)
    bilstm_model.load_state_dict(
        torch.load(
            "models/bilstm_best_model.pt",
            map_location=device
        )
    )
    bilstm_model.eval()
    test_predictions=[]
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids=batch["input_ids"].to(device)
            outputs=bilstm_model(input_ids)
            probs=torch.softmax(outputs,dim=1)

            top3=torch.topk(probs,k=3,dim=1).indices.cpu().numpy()

            for pred in top3:
                letters=[id2label[i] for i in pred]
                test_predictions.append(" ".join(letters))

    return test_predictions

