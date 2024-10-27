import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
import pickle

rv=pd.read_csv('data/reviews_text.csv')[['index','review/text']].rename({'review/text':'text'},axis=1)
train=pd.read_csv('train.csv')
rv_train=pd.merge(train[['index','score']],rv,how='left',on=['index'])
texts=list(rv_train['text'])
scores=list(rv_train['score'])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

for param in bert_model.parameters():
    param.requires_grad = False

class GetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids']).squeeze(0).to(device),
            'attention_mask': torch.tensor(inputs['attention_mask']).squeeze(0).to(device),
            'labels': torch.tensor(label).unsqueeze(0).to(device)
        }


dataset=GetDataset(texts,scores,tokenizer)
data_loader = DataLoader(dataset, batch_size=128)
class BertToVector(nn.Module):
    def __init__(self, dim=64):
        super(BertToVector, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.fc1 = nn.Linear(768, dim).to(device)
        #self.fc2 = nn.Linear(4, dim).to(device)
        self.score_layer = nn.Linear(dim, 1).to(device)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        fc_output = self.fc1(pooled_output)
        #fc_output = self.fc2(fc_output)
        score = self.score_layer(fc_output)
        return fc_output, score

model = BertToVector(dim=64)
#model.load_state_dict(torch.load('model_6_64_1.pth'))
model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam([
    {'params': model.fc1.parameters()},
    #{'params': model.fc2.parameters()},
    {'params': model.score_layer.parameters()}
], lr=3e-5)

model.train()
lst_loss=[]
for epoch in range(10):
    losses = 0
    cnt = 0
    for batch in tqdm(data_loader):
        cnt += 1
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_scores = batch['labels']
        optimizer.zero_grad()
        _, predicted_scores = model(input_ids, attention_mask)
        loss = loss_function(predicted_scores, target_scores)
        loss.backward()
        losses += loss.item()
        optimizer.step()
    model.to('cpu')
    torch.save(model.state_dict(), f'model_{epoch}_64_1.pth')
    model.to(device)
    lst_loss.append(losses/cnt)
    print(f"Epoch {epoch}, Loss: {losses / cnt}")

model.eval()
vectors = []
with torch.no_grad():
    for batch in tqdm(data_loader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        vector, _ = model(input_ids, attention_mask)
        vectors.extend(vector.cpu().numpy())

print(len(vectors), vectors[0].shape)


vectors=np.vstack(vectors)
print(vectors.shape)
with open('vecs_fc_64_1.pkl', 'wb') as f:
    pickle.dump(vectors,f)