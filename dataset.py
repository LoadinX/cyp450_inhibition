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

#%%
def local_data(preprocessed,cyp,fpname):
    molecular_description = {
        **{x.upper():'fingerprint' for x in Fingerprints},
        **{'DGL':'DGLGraph'},
        **{z.upper():'descriptor' for z in ['mordred']}
    }
    print(f'---> {molecular_description[fpname.upper()]}')
    if molecular_description[fpname.upper()] == 'fingerprint':
        order_smiles = []
        order_fp = []
        with Pool(processes=os.cpu_count()) as pool:
            for canonical_smiles,fp in pool.map(calc_fp,list(product(preprocessed.canonical_smiles,[fpname]))):
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
    elif molecular_description[fpname.upper()] == 'DGLGraph':
        data = local['canonical_smiles']
        label = local.rename(columns={cyp:'labels'})['labels'].apply(lambda x: str(int(x)))
    elif molecular_description[fpname.upper()] == 'descriptor':
        pass
    else:
        print(f'molecular description error, [{fpname}] not supported')
    return data,label

#%%
data_dir = 'material/'
class compound2dataset(Dataset):
    '''compound cid and smiles'''

    def __init__(self,smi_csv,target_csv,root_dir,output_type):
        '''
        args
            smi_csv(path or dataframe):cid,caonical_smiles
            target_csv(path or dataframe):cid,target(label/value)
            root_dir(path):usually material/
            output_type(tuple):(molecular representation,target type)
                molecular repr = fingerprint/DGLGraph/descriptor
                target type    = label/value
        '''
        self.root_dir = root_dir
        self.output_type = output_type
        smiles_frame = toframe(smi_csv)
        target_frame = toframe(target_csv)
        
        self._smiles_col = smiles_frame.columns.remove('cid')[0]
        self._target_col = target_frame.columns.remove('cid')[0]
        self.frame = pd.merge(smiles_frame,target_frame,on = 'cid',how = 'inner')
        self.num_classes = len(np.unique(np.array(pd.to_numeric(self.frame[self._target_col]))))

        
    def toframe(self,what_csv):
        if isinstance(what_csv,pd.DataFrame.type):
            what_frame = what_csv 
        elif isinstance(what_csv,str):
            what_frame = pd.read_csv(os.path.join(self.root_dir,what_csv))
        else:
            raise TypeError
        what_frame = what_frame.rename({x:x.lower() for x in what_frame.columns})
        assert len(what_frame.columns)==2
        return what_frame

    def toout(self,merged):
        if self.output_type[1] == 'label':
            self.target = torch.from_numpy(np.array(pd.to_numeric(self.frame[self._target_col],downcast='signed'))).long()
        elif self.output_type[1] == 'value':
            self.target = torch.from_numpy(np.array(pd.to_numeric(self.frame[self._target_col]))).double()
        else:
            raise TypeError
        
        self.data = calc()
        return data,target
        
    def calc(self):
        fpname = self.output_type[0]
        molecular_description = {
            **{x.upper():'fingerprint' for x in Fingerprints},
            **{'DGL':'DGLGraph'},
            **{z.upper():'descriptor' for z in ['mordred']}
        }
        print(f'---> {molecular_description[fpname.upper()]}')
        if molecular_description[fpname.upper()] == 'fingerprint':
            order_smiles = []
            order_fp = []
            with Pool(processes=os.cpu_count()) as pool:
                for canonical_smiles,fp in pool.map(calc_fp,list(product(preprocessed.canonical_smiles,[fpname]))):
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
        elif molecular_description[fpname.upper()] == 'DGLGraph':
            data = local['canonical_smiles']
            label = local.rename(columns={cyp:'labels'})['labels'].apply(lambda x: str(int(x)))
        elif molecular_description[fpname.upper()] == 'descriptor':
            pass
        else:
            print(f'molecular description error, [{fpname}] not supported')
        return data,label

    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self,index):
        