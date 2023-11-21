import pandas as pd
import torch 
from torch.utils.data import Dataset
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
CLL_feature_names = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "CD45", "CD22", "CD5", "CD19", "CD79b", "CD3", "CD81", "CD10", "CD43", "CD38"]
B_feature_names=['FSC',	'SSC'	,'CD66b',	'CD22'	,'CD19',	'CD24',	'CD10'	,'CD34'	,'CD38'	,'CD20',	'CD45']
class LogisticDataset(Dataset):
    def __init__(self, tensor_dict,df):
        super(LogisticDataset, self).__init__()
        self.dict=tensor_dict
        self.data=[[self.dict[df.iloc[i]['File'].replace('/','_')[:-4]+'.pt'],df.iloc[i]['Label']] for i in range(len(df))]
        
    def __getitem__(self, index):
        x,label=self.data[index]
        label=torch.tensor(label,dtype=torch.int64)
        x=x.to(torch.float32)
        return (x,label)
    def __len__(self):
        return len(self.data)



def DataDict(df,tensor_path):
    data_dict=dict()
    file_path_list=df['File'].values.tolist()
    for i in file_path_list:
        i=i.replace('/', '_')[:-4]+'.pt'  
        data_dict[i]=torch.load(os.path.join(tensor_path,i))
    return data_dict

 

def SaveTensor(df,dataset_path,dataset_type):
    tensor_folder='B_tensor'if dataset_type=='B-ALL' else'CLL_tensor'
    file_path_list=df['File'].values.tolist()
    for i in file_path_list:
        file_path=os.path.join(dataset_path,i)
        save_path=f"{dataset_path}/{tensor_folder}/{i.replace('/', '_')[:-4]}.pt"
        data=pd.read_csv(file_path,sep='\t')[B_feature_names]
        data_tensor=torch.tensor(data.values)
        torch.save(data_tensor,save_path)


def logic_label(df_list):
    for i in range(len(df_list)):
        df_list[i].loc[:, 'Label'] = (df_list[i]['Proportion'] > 0).astype(int)
    return df_list

def train_val_split(train_val_df, fold_nums, fold):
    percent = 1/fold_nums
    
    left = int(len(train_val_df) * (percent * fold))
    right = int(len(train_val_df) * (percent * (fold + 1)))
    
    if fold == 0:
        train = train_val_df.iloc[right:]
        val = train_val_df.iloc[:right]
    elif fold == fold_nums - 1:
        val = train_val_df.iloc[left:]
        train = train_val_df.iloc[:left]
    else:
        val = train_val_df.iloc[left:right]
        train = pd.concat([train_val_df.iloc[:left], train_val_df.iloc[right:]], axis=0)
    class_counts = train['Label'].value_counts().to_dict()
    total = len(train)
    
    # 计算每个类的比例
    class_proportions = {k: v/total for k, v in class_counts.items()}
    max_proportion = max(class_proportions.values())
    
    # 使用最大比例除以每个类别的比例来获得权重
    class_weights = {k: max_proportion/v for k, v in class_proportions.items()}
    
    # 转换为PyTorch tensor
    weight_tensor = torch.tensor([class_weights[0], class_weights[1]])
    
    return train, val, weight_tensor

    
    
if __name__ == '__main__':
    dataset_path='./dataset'
    B_ALL=pd.read_csv('./dataset/B-ALL.csv')
    CLL=pd.read_csv('./dataset/CLL.csv')
    SaveTensor(B_ALL,dataset_path,dataset_type='B-ALL')
    SaveTensor(CLL,dataset_path,dataset_type='CLL')