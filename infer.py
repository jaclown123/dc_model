"""
This is an emotion classifier based on bert-base-chinese without fine-tuning and followed by a 3-layer fully-connected
neural network. To be specific, the FNN has an input dim of 768, first hidden dim of 256 and second 64, and 4 classes.
The classifier is trained on the dataset simplifyweibo_4_moods for 20 epochs, reaching an accuracy of 57.65%.
把以下内容全部复制，调用classifier(raw_text)函数即可使用此分类器。classifier()接受一个输入raw_text，即待分类的文本；返回一个浮点数，即分类
结果，0、1、2、3分别代表喜悦、愤怒、厌恶、低落。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import BertModel, BertTokenizer


class MLP(nn.Module):
    def __init__(self, batch_size=32, feat_dim=768, n_classes=4, hidden_dim1=256, hidden_dim2=64):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, n_classes)

    def forward(self, x):
        # x is as [batch_size, feat_dim]
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        logits = self.fc3(out)
        return logits


def classify(raw_text):
    # prepare
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    bert = BertModel.from_pretrained('bert-base-chinese', num_labels=768)
    bert.to(device)
    bert.eval()

    # load model
    model = MLP()
    model.to(device)
    ckpt = torch.load("mlp_3_20epoch.pt")
    model.load_state_dict(ckpt['model'])
    print("Loaded model.")

    # predict
    model.eval()
    with torch.no_grad():
        text = tokenizer(raw_text, padding='max_length', truncation=True, return_tensors='pt')
        mask = text['attention_mask'].squeeze(1).to(device)
        input_id = text['input_ids'].squeeze(1).to(device)
        bert_output = bert(input_ids=input_id, attention_mask=mask, return_dict=True)['pooler_output']
        bert_output = bert_output.to(device)
        out_logits = model.forward(bert_output)
        model_pred = out_logits.argmax(1)
        answer = model_pred.item()
        return answer
