# calc descriptors and fingerprints of molecules

#[future] descriptors using mordred

import numpy as np
import rdkit 
from rdkit import Chem
from rdkit.Chem import AllChem,MACCSkeys
from rdkit.Avalon import pyAvalonTools
#from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs,Torsions

def calc_fp(tupled):
    # future
    #   typeerror happened in atompairs&topo
    #  TypeError: float() argument must be a string or a number, not 'IntSparseIntVect'

    (canonical_smiles,fpname) = tupled
    mol = Chem.MolFromSmiles(canonical_smiles)
    if fpname == 'MACCS':
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif fpname == 'RDKFP':
        #'RDKFP':FingerprintMols.FingerprintMol(mol),
        fp = Chem.RDKFingerprint(mol)
    elif fpname == 'MorganFP':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits = 1024)
    elif fpname == 'topo':
        fp = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
    elif fpname == 'AvalonFP':
        fp = pyAvalonTools.GetAvalonFP(mol)
    elif fpname == 'atompairs':
        fp = Pairs.GetAtomPairFingerprint(mol)
        #'PubchemFP':[int(x) for x in pcp.get_compounds(canonical_smiles,'smiles',as_dataframe=True)['cactvs_fingerprint'].tolist()[0]]
    return canonical_smiles,np.array(fp)