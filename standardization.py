import pandas as pd 
import rdkit 
from rdkit import Chem
from multiprocessing import Pool 
from standardiser import standardise
import numpy as np 
import os


def accelerated_find(tupled):
    iter,line = tupled
    if "> <PUBCHEM_COMPOUND_CID>" in line:
        cid_ = int(lines[iter+1].strip('\n'))
        return 'cid',cid_
    elif "> <PUBCHEM_OPENEYE_CAN_SMILES>" in line:
        mol = Chem.MolFromSmiles(lines[iter+1].strip('\n'))
        try:
            parent =  standardise.run(mol)
            smiles_ = Chem.MolToSmiles(parent)
        except Exception:
            smiles_ = np.nan
        return 'smiles',smiles_
    else:
        return 'None',np.nan

def get_standard(sdf_path = 'material/CID.sdf',out_path = 'material/standard_compounds.csv'):
    print('compounds standarization on going...')
    cid = []
    canonical_smiles = []
    global lines 
    lines = open(sdf_path,'r').readlines()

    with Pool(processes=os.cpu_count()) as pool:
        for t,r in pool.map(accelerated_find,list(enumerate(lines))):
            if t == 'cid':
                cid.append(r)
            elif t == 'smiles':
                canonical_smiles.append(r)
            else:
                pass

    result = pd.DataFrame()
    result['cid'] = cid 
    result['canonical_smiles'] = canonical_smiles
    count_nan = result['canonical_smiles'].isna().sum()
    result = result.dropna()
    print(f'standization: unable to standize : {count_nan}, kept {len(result)}')
    result.to_csv(out_path,index = False)
    return result


