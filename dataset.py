#%%
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np 
import pandas as pd
import os 
from calc_descri import calc_fp

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# supporting:
Fingerprints = ['MACCS','RDKFP','MorganFP','AvalonFP']

#%%
data_dir = 'material'
class compound2dataset(Dataset):
    '''compound cid and smiles'''

    def __init__(self,smi_csv,target_csv,data_dir = 'material',output_type):
        '''
        args
            smi_csv(path:cyp_smiles.csv or dataframe):cid,caonical_smiles
            target_csv(path:cyp_target.csv or dataframe):cid,target(label/value)

            output_type(tuple):(molecular representation,target type)
                molecular repr = fingerprint/DGLGraph/descriptor
                target type    = label/value
        '''
        self.root_dir = os.getcwd()
        self.data_dir = os.path.join(self.root_dir,data_dir)
        self.output_type = output_type
        smiles_frame = transframe(smi_csv)
        target_frame = numeric(transframe(target_csv))
        self.num_classes = len(np.unique(self.target))

        
    def transframe(self,what_csv):
        if isinstance(what_csv,pd.DataFrame.type):
            what_frame = what_csv 
        elif isinstance(what_csv,str):
            what_frame = pd.read_csv(os.path.join(self.data_dir,what_csv))
        else:
            raise TypeError
        what_frame = what_frame.rename({x:x.lower() for x in what_frame.columns})
        assert len(what_frame.columns)==2
        return what_frame

    def numeric(self,frame):
        col_name = frame.columns.remove('cid')[0]
        if self.output_type[1] == 'label':
            self.target = pd.to_numeric(frame[col_name],downcast = 'integer').tolist()
        elif self.output_type[1] == 'value':
            self.target = pd.to_numeric(frame[col_name],downcast = 'float').tolist()
        else:
            raise TypeError
        frame[col_name] = self.target
        return frame
        
    def calc(self,preprocessed):
        decri = self.output_type[0]
        molecular_description = {
            **{x.upper():'fingerprint' for x in Fingerprints},
            **{'DGL':'DGLGraph'},
            **{z.upper():'descriptor' for z in ['mordred']}
        }
        print(f'---> {molecular_description[decri.upper()]}')
        if molecular_description[decri.upper()] == 'fingerprint':
            order_smiles = []
            order_fp = []
            with Pool(processes=os.cpu_count()) as pool:
                for canonical_smiles,fp in pool.map(calc_fp,list(product(preprocessed.canonical_smiles,[decri]))):
                    order_smiles.append(canonical_smiles)
                    order_fp.append(fp)
            newdf = pd.DataFrame()
            newdf['canonical_smiles'] = order_smiles
            newdf['fp'] = order_fp
            newdf[cyp] = np.nan
            newdf.update(preprocessed)
            #有标签和能够标准化的分子之间有差异
            newdf = newdf.dropna()
            label = newdf.rename(columns={cyp:'labels'})['labels'].apply(lambda x:int(x))
            data = np.array(newdf['fp'].tolist())
        elif molecular_description[decri.upper()] == 'DGLGraph':
            data = local['canonical_smiles']
            label = local.rename(columns={cyp:'labels'})['labels'].apply(lambda x: str(int(x)))
        elif molecular_description[decri.upper()] == 'descriptor':
            pass
        else:
            print(f'molecular description error, [{decri}] not supported')
        return data,label

    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self,index):
        