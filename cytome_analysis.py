import pandas as pd
from model import *
import torch
import seaborn as sns
import matplotlib.pyplot as plt

CLL_feature_names = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "CD45", "CD22", "CD5", "CD19", "CD79b", "CD3", "CD81", "CD10", "CD43", "CD38"]
B_feature_names=['FSC',	'SSC'	,'CD66b',	'CD22'	,'CD19',	'CD24',	'CD10'	,'CD34'	,'CD38'	,'CD20',	'CD45']

device = torch.device("cuda")

def generate_heatmap_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    required_columns = {'Start Index', 'End Index', 'Probability Change'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(df[['Probability Change']].T, fmt=".2f", cmap="YlGnBu")
    plt.title("Probability Change Heatmap")
    png_path = f'{csv_file_path[:-4]}_heatmap.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    return png_path

def cytome_analysis(model,data,begin,end,num_parts,file_name):
    if end>len(data):
        raise ValueError("end is greater than the length of data")
    elif end-begin<num_parts:
        raise ValueError("num_parts is greater than the length of block")
    # Calculate initial probability
    initial_prob = F.softmax(model(data.unsqueeze(0)).cpu().detach(),dim=1)[0,1]
    # List used to store probability changes
    results = []
    part_size = (end - begin) // num_parts

    for i in range(begin, end, part_size):
        # Make sure not to exceed the end of the data
        if i + part_size > end:
            new_x = data[:i]
        else:
            new_x = torch.cat([data[:i],data[i + part_size:]],dim=0)
            
        new_prob = F.softmax(model(new_x.unsqueeze(0)).cpu().detach(),dim=1)[0,1]
        prob_change = initial_prob-new_prob
        results.append({'Start Index': i, 'End Index': i + part_size, 'Probability Change': prob_change.item()})
    csv_path=f'{file_name[:-4]}_analyzed.csv'
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    png_path=generate_heatmap_from_csv(csv_path)
    
    return float(initial_prob), csv_path, png_path

def run_model(blood_cancer_type: str, file_name: str, begin: int = None, end: int = None, num_parts: int = 100) -> torch.Tensor:
    #blood_caner_type:CLL or B-ALL
    #file_name: csv/txt file path uploaded by the user
    #begin: Single cell initial subscripts for interpretability analysis
    #end: Single cell end index for interpretability analysis
    #num_parts: Number of single cell subsets used for interpretability analysis
    
    model_path='./pretrained_model/CLL.ckpt' if blood_cancer_type=='CLL' else './pretrained_model/B-ALL.ckpt'
    #load model
    model=Model(None,blood_cancer_type).to(device)
    checkpoint = torch.load(model_path,map_location=device)
    del checkpoint['state_dict']['ploy_loss.CELoss.weight']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    #read data
    df=pd.read_csv(file_name,sep='\t')
    df=df[CLL_feature_names] if blood_cancer_type=='CLL' else df[B_feature_names]
    data=torch.tensor(df.values).to(torch.float32).to(device)
    #run model
    if begin is None or end is None:
        begin = 0
        end = len(data)
    prob, csv_path, png_path = cytome_analysis(model,data,begin,end,num_parts,file_name)
    return prob, csv_path, png_path

if __name__=='__main__':
    import time
    t1=time.time()
    run_model(blood_cancer_type='B-ALL',file_name='./149.fcs.txt')
    t2=time.time()
    print(t2-t1)