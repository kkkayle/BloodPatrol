import pandas as pd
from torch.utils.data import DataLoader
from utils.dataset import *
from model import *
import torch
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
import random
import argparse
import shutil

#parser.add_argument('--',type=,default=)
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--dataset',type=str,default='CLL',choices=['B-ALL','CLL'])
parser.add_argument('--fold_nums',type=int,default=5)
parser.add_argument('--mode',type=str,default='train',choices=['train','test'])
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.__version__)


set_seed(args.seed)
tensor_path=f'./dataset/{args.dataset}_tensor'
data_df=pd.read_csv(f'./dataset/{args.dataset}.csv')
data_df=data_df.sample(frac=1,random_state=args.seed).reset_index(drop=True)

test_df=data_df.iloc[:int(len(data_df)*0.2)].copy()
train_val_df=data_df.iloc[int(len(data_df)*0.2):].copy()
model_path = [f"checkpoint_model{'-v' + str(i) if i else ''}.ckpt" for i in range(args.fold_nums)]

if args.dataset=='B-ALL':
    train_val_df,test_df=logic_label([train_val_df,test_df]) 


if args.mode=='train':
    if os.path.exists('./checkpoint') and os.path.isdir('./checkpoint'):
        shutil.rmtree('./checkpoint')
    for fold in range(args.fold_nums):
        train_df, val_df,class_weight = train_val_split(train_val_df,fold_nums=args.fold_nums, fold=fold)
        
        train_dict=DataDict(train_df,tensor_path)
        val_dict=DataDict(val_df,tensor_path)
        test_dict=DataDict(test_df,tensor_path)
        
        train_dataset=LogisticDataset(train_dict,train_df)
        val_dataset=LogisticDataset(val_dict,val_df)
        test_dataset=LogisticDataset(test_dict,test_df)

        train_dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=0)
        val_dataloader=DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=0)
        test_dataloader=DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=0)

        model=Model(class_weight,args.dataset)
        
        checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_f1",
        mode="max",
        dirpath='./checkpoint',
        filename=f"checkpoint_model",
        every_n_epochs=1,
        )
        
                    

        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=50,accumulate_grad_batches=4, logger=pl.loggers.CSVLogger('logs'),
                                enable_checkpointing=True, callbacks=[checkpoint_callback,pl.callbacks.StochasticWeightAveraging(swa_lrs=0.0001,annealing_epochs=1,swa_epoch_start=0.3)])

        trainer.fit(model, train_dataloader, val_dataloader)
        checkpoint = torch.load(os.path.join('./checkpoint',model_path[fold]))

        model.load_state_dict(checkpoint['state_dict'])
        trainer.test(model,test_dataloader)
else:
    test_dict=DataDict(test_df,tensor_path)
    test_dataset=LogisticDataset(test_dict,test_df)
    test_dataloader=DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=0)
    model_list=[]
    for path in model_path:
        model=Model(None,args.dataset)
        checkpoint = torch.load(os.path.join('./checkpoint',path))
        del checkpoint['state_dict']['ploy_loss.CELoss.weight']
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model_list.append(model)
    out_list=[]
    label_list=[]
    from tqdm import tqdm
    for x,label in tqdm(test_dataloader):
        x=x.cuda()
        out_sum=0
        for model in model_list:
            model=model.cuda()
            out = model(x)
            out_sum+=out
        out_sum/=len(model_list)
        out_sum=out_sum.detach().cpu().numpy()
        label=label.detach().cpu().numpy()
        out_list.append(out_sum)
        label_list.append(label)
    out_tensor=torch.tensor(out_list).squeeze(1)
    label_tensor=torch.tensor(label_list)
    metrics = model_list[0].cal_metrics(out_tensor,label_tensor)
    
    for key,value in metrics.items():
        print(f"test_{key}", value)