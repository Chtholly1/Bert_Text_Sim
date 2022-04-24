import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import scipy.stats
from tqdm import tqdm
from conf.config import *

def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation

def compute_loss(y_pred0, y_pred1, y_pred2, lamda=0.05):
    y_pred = torch.cat((y_pred1, y_pred2), 0)
    #idxs = torch.arange(0,y_pred.shape[0],device='cuda')
    y_true = torch.arange(0, y_pred0.shape[0], device='cuda')
    similarities = F.cosine_similarity(y_pred0.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    #print(similarities.shape)
    #torch自带的快速计算相似度矩阵的方法
    similarities = similarities / lamda
    #论文中除以 temperature 超参 0.05
    loss = F.cross_entropy(similarities, y_true)
    #exit()
    return torch.mean(loss), similarities


def test_acc(dataloader,model):
    correct = 0
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            input_ids0 = data['input_ids'][:,0].to(device)
            attention_mask0 = data['attention_mask'][:,0].to(device)
            token_type_ids0 = data['token_type_ids'][:,0].to(device)
            input_ids1 = data['input_ids'][:,1].to(device)
            attention_mask1 = data['attention_mask'][:,1].to(device)
            token_type_ids1 = data['token_type_ids'][:,1].to(device)
            input_ids2 = data['input_ids'][:,2].to(device)
            attention_mask2 = data['attention_mask'][:,2].to(device)
            token_type_ids2 = data['token_type_ids'][:,2].to(device)
            pred0 = model(input_ids0, attention_mask0, token_type_ids0)
            pred1 = model(input_ids1, attention_mask1, token_type_ids1)
            pred2 = model(input_ids2, attention_mask2, token_type_ids2)
            loss, similarities = compute_loss(pred0, pred1, pred2)
            _, out_classes = similarities.max(dim=-1)
            out_classes_new = out_classes.view(-1,1)
            targets_new = torch.arange(0, input_ids0.shape[0], device='cuda')
            temp_correct = (out_classes_new == targets_new).sum()
            correct += temp_correct/input_ids0.shape[0]
    acc = correct/len(dataloader)
    return loss, acc

def test_auc(dataloader,model):
    all_auc = 0
    pred_all = []
    targets_all = []
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            input_ids0 = data['input_ids'][:,0].to(device)
            attention_mask0 = data['attention_mask'][:,0].to(device)
            token_type_ids0 = data['token_type_ids'][:,0].to(device)
            input_ids1 = data['input_ids'][:,1].to(device)
            attention_mask1 = data['attention_mask'][:,1].to(device)
            token_type_ids1 = data['token_type_ids'][:,1].to(device)
            input_ids2 = data['input_ids'][:,2].to(device)
            attention_mask2 = data['attention_mask'][:,2].to(device)
            token_type_ids2 = data['token_type_ids'][:,2].to(device)
            pred0 = model(input_ids0, attention_mask0, token_type_ids0)
            pred1 = model(input_ids1, attention_mask1, token_type_ids1)
            pred2 = model(input_ids2, attention_mask2, token_type_ids2)
            pos_pred = F.cosine_similarity(pred0, pred1, dim=-1)
            neg_pred = F.cosine_similarity(pred0, pred2, dim=-1)
            pred = torch.cat((pos_pred,neg_pred)).cpu().tolist()
            targets_new = torch.cat((torch.ones(input_ids0.shape[0]), torch.zeros(input_ids0.shape[0]))).tolist()
            pred_all.extend(pred)
            targets_all.extend(targets_new)
    auc = roc_auc_score(targets_all, pred_all)
    return auc

def test(test_data,model):
    traget_idxs, source_idxs, label_list = test_data.get_data()
    with torch.no_grad():
        traget_input_ids = traget_idxs['input_ids'].to(device)
        traget_attention_mask = traget_idxs['attention_mask'].to(device)
        traget_token_type_ids = traget_idxs['token_type_ids'].to(device)
        traget_pred = model(traget_input_ids,traget_attention_mask,traget_token_type_ids)

        source_input_ids = source_idxs['input_ids'].to(device)
        source_attention_mask = source_idxs['attention_mask'].to(device)
        source_token_type_ids = source_idxs['token_type_ids'].to(device)
        source_pred = model(source_input_ids,source_attention_mask,source_token_type_ids)

        similarity_list = F.cosine_similarity(traget_pred,source_pred)
        similarity_list = similarity_list.cpu().numpy()
        label_list = np.array(label_list)
        corrcoef = compute_corrcoef(label_list,similarity_list)
    return corrcoef

def train(dataloader, test_dataloader, model, optimizer):
    model.train()
    size = len(dataloader.dataset)
    max_acc = 0
    for batch, data in enumerate(dataloader):
        input_ids0 = data['input_ids'][:,0].to(device)
        attention_mask0 = data['attention_mask'][:,0].to(device)
        token_type_ids0 = data['token_type_ids'][:,0].to(device)
        input_ids1 = data['input_ids'][:,1].to(device)
        attention_mask1 = data['attention_mask'][:,1].to(device)
        token_type_ids1 = data['token_type_ids'][:,1].to(device)
        input_ids2 = data['input_ids'][:,2].to(device)
        attention_mask2 = data['attention_mask'][:,2].to(device)
        token_type_ids2 = data['token_type_ids'][:,2].to(device)
        pred0 = model(input_ids0, attention_mask0, token_type_ids0)
        pred1 = model(input_ids1, attention_mask1, token_type_ids1)
        pred2 = model(input_ids2, attention_mask2, token_type_ids2)
        loss,_ = compute_loss(pred0, pred1, pred2)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * int(len(input_ids0))
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            model.eval()
            #corrcoef = test(testdata,model)
            loss, acc = test_acc(test_dataloader, model)
            model.train()
            print(f"corrcoef_test: {acc:>4f}")
            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(),save_path)
                print(f"Higher acc: {(max_acc):>4f}%, Saved PyTorch Model State to model.pth")

