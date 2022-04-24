#coding:utf-8
from functools import partial
import torch
import random
import numpy as np
import scipy as sp
import scipy.optimize

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class FGM(object):
    def __init__(self, model, emb_name, epsilon=1):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD(object):
    def __init__(self, model, emb_name):
        self.model = model
        self.emb_name = emb_name
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

def setup_seed(seed=2022):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def acc(x, y):
    x = np.array(x)
    y = np.array(y)
    score = np.sum(x ==  y)
    return score/x.shape[0]

def CE(X, Y):
    cross_loss = 0
    for idx, x in enumerate(X):
        cross_loss += Y[idx]*np.log(X[idx]+1e-6) + (1-Y[idx])*np.log(1-X[idx]+1e-6)
    return cross_loss

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            else:
                X_p[i] = 1
        print(X_p)
        print(y)
        ll = acc(X_p, y)
        return -ll

    def mult_class_loss(self, coef, X, y):
        X_p = np.copy(X)
        coef_np = np.array(coef)#.reshape(-1,1) 
        X_p = X_p * coef_np
        pred = np.argmax(X_p, axis=1)        
        score = acc(pred, y)
        return -score

    def test(self, x):
        res = 0.8*np.log(x) + 0.2*np.log(1-x)
        return -res

    def fit(self, X, y):
        #initial_coef = [1]*X.shape[1]
        #self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        #self.coef_ = scipy.optimize.minimize(self._kappa_loss, initial_coef, (X,y), method='Nelder-Mead')
        initial_x = 0.5
        self.coef_ = scipy.optimize.minimize(self.test, initial_x, method='Nelder-Mead')
        print(self.coef_)

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            else:
                X_p[i] = 1
        return X_p

    def coefficients(self):
        return self.coef_['x']

if __name__ == '__main__':
    c = OptimizedRounder()
    x = [0.01*x for x in range(100)]
    y = [0]*30 + [1]*70
    c.fit(x, y)
    coef = c.coefficients()

