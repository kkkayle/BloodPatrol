import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
class WeightNetwork(nn.Module):
    def __init__(self, filed_size, reduction_ratio=3):
        super(WeightNetwork, self).__init__()
        self.reduction_size = max(1, filed_size // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(filed_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, filed_size, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        Z = torch.mean(inputs, dim=-1, out=None)
        A = self.excitation(Z)
        V = torch.mul(inputs, torch.unsqueeze(A, dim=-1))
        return V



class ResidualConv1D(nn.Module):
    def __init__(self,dim_1,dim_2,dim_3):
        super(ResidualConv1D, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=dim_1, out_channels=dim_2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Second convolution layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=dim_2, out_channels=dim_2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Third convolution layer
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=dim_2, out_channels=dim_3, kernel_size=3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1) + out1
        out3 = self.conv3(out2) 
        return out3



class PolyLoss(nn.Module):
    def __init__(self, DEVICE,weight_loss=None, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_loss, reduction='none')
        self.epsilon = epsilon
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        batch_size=labels.shape[0]
        one_hot = torch.zeros((batch_size, 2), device=self.DEVICE).scatter_(
            1, labels.to(torch.int64), 1)
        pt = torch.sum(one_hot * F.softmax(predicted, dim=1), dim=-1)
        ce = self.CELoss(predicted, labels)
        poly1 = ce + self.epsilon * (1-pt)
        return torch.mean(poly1)


class Model(pl.LightningModule):
    def __init__(self,class_weight,dataset,lr=0.0001,epoch_size=185):
        super().__init__()
        self.save_hyperparameters()
        self.in_feats_dim=11 if dataset=='B-ALL' else 16
        self.nhead=11 if dataset=='B-ALL' else 4
        self.transformer=nn.TransformerEncoderLayer(d_model=self.in_feats_dim,nhead=self.nhead)
        self.encoder = nn.TransformerEncoder(self.transformer,num_layers=1)
        self.conv_pool=ResidualConv1D(self.in_feats_dim,self.in_feats_dim,4)
        
        self.step_size_up=5
        self.decoder=nn.Sequential(
            nn.Linear(400,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,2),
        )
        
        self.ploy_loss=PolyLoss(DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu"),weight_loss=class_weight)
        self.val_out=[]
        self.test_out=[]
        self.desired_output_dim=100
        self.layer_norm=nn.LayerNorm(self.in_feats_dim)
        self.ts_weight=WeightNetwork(100)


    def forward(self,x):
        x=self.layer_norm(x)
        encode=self.encoder(x)
        encode_pooled=self.max_pooling(encode)
        ts_pooled=self.ts_weight(encode_pooled)*0.5+encode_pooled*0.5
        pooled=self.conv_pool(ts_pooled.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.decoder(pooled.reshape(x.shape[0],-1))
        return out

    def on_validation_epoch_end(self):
        """
        'AUC': auc,
        'AUPR': aupr,
        'Accuracy': acc,
        'Recall': recall,
        'Precision': precision,
        'F1': f1
        """
        out=torch.tensor([i[0] for i in self.val_out])
        label=torch.tensor([i[1] for i in self.val_out])
        metrics = self.cal_metrics(out, label)
        f1 = metrics['F1']
        self.log("val_f1", f1, prog_bar=True, logger=True, on_epoch=True)
        self.val_out.clear()
        
    def max_pooling(self, x):
        x = x.permute(0, 2, 1)
        pooled = F.adaptive_max_pool1d(x, output_size=self.desired_output_dim)
        pooled = pooled.permute(0, 2, 1)
        return pooled
    

    def loss_function(self, pred, target):
        loss=self.ploy_loss(pred,target)
        #loss = F.cross_entropy(pred, target)
        return loss   
    
    
    def cal_metrics(self,pred, label):

        pred=F.softmax(pred,dim=1)
        pred_label = pred.argmax(axis=1)


        acc = accuracy_score(label, pred_label)
        recall = recall_score(label, pred_label)
        precision = precision_score(label, pred_label)
        f1 = f1_score(label, pred_label)

        try:
            auc = roc_auc_score(label, pred[:, 1])
        except:
            auc=0
        aupr = average_precision_score(label, pred[:, 1])

        return {
            'AUC': auc,
            'AUPR': aupr,
            'Accuracy': acc,
            'Recall': recall,
            'Precision': precision,
            'F1': f1
        }



    def training_step(self, train_batch, batch_idx):
        x, label = train_batch
        out = self(x)
        label_onehot = F.one_hot(label, 2).float().cuda()
        loss = self.loss_function(out, label_onehot)
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, label = val_batch
        out = self(x)
        out=out.detach().cpu().numpy().tolist()
        label=label.detach().cpu().numpy().tolist()
        for i in range(len(out)):
            self.val_out.append([out[i],label[i]])




    def test_step(self, test_batch, batch_idx):
        x, label = test_batch
        out = self(x)
        out = out.detach().cpu().numpy().tolist()
        label = label.detach().cpu().numpy().tolist()
        for i in range(len(out)):
            self.test_out.append([out[i],label[i]])

    def on_test_epoch_end(self):
        out = torch.tensor([i[0] for i in self.test_out])
        label = torch.tensor([i[1] for i in self.test_out])
        metrics = self.cal_metrics(out, label)


        for key,value in metrics.items():
            self.log(f"test_{key}", value)
        
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),weight_decay=1e-4)
        """
        max_lr=self.hparams.lr
        base_lr = max_lr/5.0
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
            base_lr=base_lr,max_lr=max_lr,
            step_size_up=2*self.hparams.epoch_size,cycle_momentum=False)
        self.print("set lr = "+str(max_lr))
        """
        #return ([optimizer],[scheduler])
        return ([optimizer])