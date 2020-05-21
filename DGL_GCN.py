from collections import namedtuple

from rdkit import Chem
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score, confusion_matrix

import dgl
from dgl import DGLGraph
import dgl.function as fn 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset,TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim 

import os 
import copy 
import pandas as pd
import numpy as np 
import networkx as nx 
from multiprocessing import Pool
import matplotlib.pyplot as plt 
from standardization import get_standard
import time
import warnings
warnings.filterwarnings('ignore')

global device 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#currently,lab own 1XGTX2080Ti, [future] add DataPara

#reproductbility
np.random.seed(30191375)
torch.manual_seed(30191375)
torch.cuda.manual_seed(30191375)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.multiprocessing.set_start_method('spawn',force = True)


#%%
#define elements
ELEM_LIST = ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','Al','I','B','K','Se','Zn','H','Cu','Mn','unknown']
ATOM_FDIM = len(ELEM_LIST)+6+5+1
MAX_ATOMNUM = 60
BOND_FDIM = 5
MAX_NB = 10


## original code 

#%% generate graph of molecular structure, source:dgl function tree


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]
# Note that during graph decoding they don't predict stereochemistry-related
# characteristics (i.e. Chiral Atoms, E-Z, Cis-Trans).  Instead, they decode
# the 2-D graph first, then enumerate all possible 3-D forms and find the
# one with highest score.
'''
def atom_features(atom):
    return (torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + [atom.GetIsAromatic()]))
'''
def atom_features(atom):
    return (onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + [atom.GetIsAromatic()])
def bond_features(bond):
    bt = bond.GetBondType()
    return (torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]))
def mol2dgl(mols):
    cand_graphs = []
    n_nodes = 0
    n_edges = 0
    bond_x = []
    for mol in mols:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()
        g = DGLGraph()        
        nodeF = []
        for i, atom in enumerate(mol.GetAtoms()):
            assert i == atom.GetIdx()
            nodeF.append(atom_features(atom))
        g.add_nodes(n_atoms)
        bond_src = []
        bond_dst = []
        for i, bond in enumerate(mol.GetBonds()):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            begin_idx = a1.GetIdx()
            end_idx = a2.GetIdx()
            features = bond_features(bond)
            bond_src.append(begin_idx)
            bond_dst.append(end_idx)
            bond_x.append(features)
            bond_src.append(end_idx)
            bond_dst.append(begin_idx)
            bond_x.append(features)
        g.add_edges(bond_src, bond_dst)
        g.ndata['h'] = torch.Tensor(nodeF)
        cand_graphs.append(g)
    return cand_graphs  

def mol2dgl_single(mol):
    n_nodes = 0
    n_edges = 0
    bond_x = []
    
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    g = DGLGraph()        
    nodeF = []
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        nodeF.append(atom_features(atom))
    g.add_nodes(n_atoms)
    bond_src = []
    bond_dst = []
    for i, bond in enumerate(mol.GetBonds()):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        begin_idx = a1.GetIdx()
        end_idx = a2.GetIdx()
        features = bond_features(bond)
        bond_src.append(begin_idx)
        bond_dst.append(end_idx)
        bond_x.append(features)
        bond_src.append(end_idx)
        bond_dst.append(begin_idx)
        bond_x.append(features)
    g.add_edges(bond_src, bond_dst)
    g.ndata['h'] = torch.Tensor(nodeF)
    return g

message = fn.copy_src(src = 'h', out = 'm')

def reduce(nodes):
    """对所有邻节点节点特征求平均并覆盖原本的节点特征。"""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}
#%% definition of GCN framework

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(message, reduce)
        g.apply_nodes(func=self.apply_mod)
        h =  g.ndata.pop('h').to(device)
        #print('h_shape',h.shape)
        return h
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.layers = nn.ModuleList([GCN(in_dim, hidden_dim, F.relu),
                                    GCN(hidden_dim, hidden_dim//2, F.relu),
                                    #GCN(hidden_dim,hidden_dim//2,F.relu)
                                    ])
        self.classify = nn.Linear(hidden_dim//2, n_classes)
        self.simple = nn.Sequential(
            nn.Linear(hidden_dim//2,hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2,hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4,n_classes*2)
        )
        self.dropout = nn.Sequential(
            nn.Linear(hidden_dim//2,hidden_dim//2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_dim//2,hidden_dim//4),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_dim//4,n_classes*2)
        )
    def forward(self, g):
        #h = g.ndata['h']
        h = g.in_degrees().view(-1,1).float().to(device)
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        graph_repr = dgl.mean_nodes(g, 'h')
        pred = self.single(graph_repr)
        return pred
        #return  self.classify(graph_repr)
        #out = self.classify(graph_repr)
        #return nn.functional.log_softmax(out,dim=1)
        #res = self.out(graph_repr)
        #return F.sigmoid(res)
        #out = self.out(graph_repr)
        #return out

class DNN(nn.Module):
    def __init__(self,in_dim,layers,out_dim):
        super(DNN,self).__init__()


# dataloader

def smi2mol(string):
    mol = Chem.MolFromSmiles(string[:-1])
    graph = mol2dgl_single(mol)
    label = int(string[-1])
    return (graph,label)
    
def df2dataset(df):
    'dataframe must contain columns: canonical_smiles,labels'
    df['string'] = df['canonical_smiles']+df['labels'].apply(lambda x:str(x))
    graphs = []
    labels = []
    with Pool(processes = os.cpu_count()) as pool:
        for (g,l) in pool.map(smi2mol,df['string'].tolist()):
            graphs.append(g)
            labels.append(l)
    return graphs,labels

class dataset(Dataset):
    def __init__(self,graphs,labels):
        self.data = graphs
        self.nplabel = np.array(pd.to_numeric(labels,downcast='signed'))
        self.label = torch.from_numpy(self.nplabel).long()
        self.num_classes = self.nplabel.ndim

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)

def collate(samples):
    assert isinstance(samples,list) == True
    assert isinstance(samples[0],tuple) == True
    assert len(samples[0]) == 2

    graphs,labels = map(list,zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph,torch.tensor(labels)

def loss_plot():
    losses = pd.read_csv('epoch_losses/losses.csv').values.tolist()
    for cyp,iter,epoch_losses in losses:
        plt.plot(epoch_losses,c = 'b')
        fig = plt.gcf()
        fig.savefig(f'epoch_losses/{cyp}_cv{iter}.tiff')
    print('plotting done')

def load_checkpoint(model,optimizer,checkpoint_PATH):
    model_checkpoint = torch.load(checkpoint_PATH)
    model.load_state_dict(model_checkpoint['state_dict'])
    if optimizer == None:
        return model
    else:
        optimizer.load_state_dict(model_checkpoint['optimizer'])
        return model,optimizer

def evaluate(model, data,label):
    '[future] convert data,label to dataloader(one batch),return auc,acc,se,sp'
    'data: pd.Series: Canonical_Smiles need process to calc graph'
    eval_graphs,eval_labels = df2dataset(pd.concat([data,label],axis=1))
    evalset = dataset(eval_graphs,eval_labels)
    evalloader = DataLoader(evalset,batch_size=len(eval_labels),shuffle= False,drop_last= False, collate_fn= collate,num_workers=0)

    model = model.to(device)    
    model.eval()
    for eval_graph,eval_label in evalloader:
        with torch.no_grad():
            eval_graph,eval_label = eval_graph.to(device),eval_label.to(device)
            eval_graph.set_e_initializer(dgl.init.zero_initializer)
            eval_graph.set_n_initializer(dgl.init.zero_initializer)
            proba = model(eval_graph)
            proba = proba.to(device)
            _, indices = torch.max(proba, dim=1)
            correct = torch.sum(indices == eval_label)
            correct = correct.to(device)
    return correct.item() * 1.0 / len(eval_label)

if __name__ == "__main__":
    
    start = time.time()
    print(start)
    # load data
    origin = pd.read_csv('material/extracted_1851.csv')


    if 'standard_compounds.csv' in os.listdir(os.getcwd()+'/material'):
        standard_result = pd.read_csv('material/standard_compounds.csv')
    else:
        from standardization import get_standard
        standard_result = get_standard()

    df = origin.dropna(axis = 0,subset = ['CID'])  # drop compounds without pubchem CID
    df['cid'] = df['CID']
    df['canonical_smiles'] = np.nan
    df.update(standard_result[['cid','canonical_smiles']])
    df = df.dropna(axis = 0,subset = ['canonical_smiles']) # drop compounds fail in standarization

    cyps = []
    for col in df.columns:
        if 'cyp' in col:
            cyps.append(col)
    mix = df[cyps+['canonical_smiles']]

    res_dict = {x:[] for x in ['endpoint','fp','method','AUC','SE','SP','ACC']}

    split = 'random' # [future] 'random','kmean','cluster',etc
    EPOCH = 25

    try:
        os.mkdir('models')
    except Exception:
        print(f'dir <models> be existed')

    try:
        os.mkdir('epoch_losses')
    except Exception:
        print(f'dir <epoch_losses> be existed')    

    cv_res_columns = ['cyp','best_model','AUC','SE','SP','ACC']
    cv_res_lines = []
    model_eval_lines = []
    epoch_loss_graph = []
    for cyp in cyps:
        print('------------------------------------')
        local = mix[[cyp]+['canonical_smiles']].dropna() # drop compounds without label in exact cyp enzyme
        local = local.drop_duplicates('canonical_smiles',keep = 'first')
        print(f'[{cyp}] positive/negative = {int(local[cyp].sum())}/{int(len(local)-local[cyp].sum())}')
        data = local['canonical_smiles']
        label = local.rename(columns={cyp:'labels'})['labels'].apply(lambda x: str(int(x)))

        # for cross validation in deel learning:
        #   split data into train/valid/test = 8:1:1
        #   hyperparameter searching via trainset training,validset tuning, get a model_state_dict(tuned params)
        #   暂时这里没有做交叉验证
        if split == 'random':
            X_train,X_test,y_train,y_test = train_test_split(data,label,test_size = 0.2,random_state = 30191375)
        
        ## cross validation
        cv_dict = {x:[] for x in ['cyp','model','AUC','SE','SP','ACC']}
        kfold = StratifiedKFold(n_splits = 5,shuffle=True,random_state = 100)
        for iter,(train,test) in enumerate(kfold.split(X_train,y_train)):

            print(f'loading {cyp}_cv{iter} trainset...')
            train_graphs,train_labels = df2dataset(pd.concat([X_train.iloc[train],y_train.iloc[train]],axis =1))
            valid_graphs,valid_labels = df2dataset(pd.concat([X_train.iloc[test],y_train.iloc[test]],axis =1))
            trainset = dataset(train_graphs,train_labels)
            validset = dataset(valid_graphs,valid_labels)
            trainset_loader =  DataLoader(trainset,batch_size = 256, shuffle= True,drop_last= False, collate_fn= collate,num_workers=os.cpu_count())
            validset_loader = DataLoader(validset,batch_size = len(valid_labels),collate_fn=collate)
            # create model 
            model = Classifier(1,256,trainset.num_classes)
            loss_func = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(),lr = 0.001)
            model.train()

            epoch_losses = []
            valid_losses = []
            for epoch in range(EPOCH):
                epoch_loss = 0
                for i,(bg,label) in enumerate(trainset_loader):
                    bg.set_e_initializer(dgl.init.zero_initializer)
                    bg.set_n_initializer(dgl.init.zero_initializer)
                    pred = model(bg)
                    loss = loss_func(pred,label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.detach().item()
                epoch_loss /= (i+1)
                print(epoch_loss) ##########################
                epoch_losses.append(epoch_loss)

                model.eval()
                with torch.no_grad():
                    for vbg,vlabel in validset_loader:
                        valid_pred = model(vbg)


                valid_loss = loss_func(valid_pred,vlabel).item()
                valid_losses.append(valid_loss)

                valid_pred = torch.max(valid_pred,1)[1].tolist()
                if (epoch+1) % 5 == 0:
                    print(f'Epoch {epoch+1}: train_loss {epoch_loss:.4f},valid_loss {valid_loss:.4f}')

            epoch_loss_graph.append([cyp,iter,epoch_losses,valid_losses])
            torch.save({'epoch': EPOCH + 1, 'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()},f'models/checkpoint_{cyp}_cv{iter}.pth.tar')

            acc = accuracy_score(valid_labels,valid_pred)
            rec = recall_score(valid_labels,valid_pred,pos_label = 1, average = 'binary')
            auc = roc_auc_score(valid_labels,valid_pred)
            matrix = confusion_matrix(valid_labels,valid_pred,labels = [0,1])
            sp = matrix[0,0]*1.0/(matrix[0,0]+matrix[0,1])

            cv_dict['cyp'].append(cyp)
            cv_dict['model'].append(iter)
            cv_dict['AUC'].append(auc)
            cv_dict['SE'].append(rec)
            cv_dict['SP'].append(sp)
            cv_dict['ACC'].append(acc)

        cv_df = pd.DataFrame(cv_dict)
        best_model_index = cv_df.sort_values(by = 'AUC',ascending = False).iloc[0]['model']
        cv_res_lines.append([cyp,best_model_index]+cv_df.mean()[['AUC','SE','SP','ACC']].tolist())
        
        best_model_checkpoint = torch.load(f'models/checkpoint_{cyp}_cv{best_model_index}.pth.tar')
        best_model = Classifier(1,256,trainset.num_classes)
        best_model.load_state_dict(best_model_checkpoint['state_dict'])
        best_model.eval()

        test_graphs,test_labels = df2dataset(pd.concat([X_test,y_test],axis =1))
        testset = dataset(test_graphs,test_labels)
        testset_loader = DataLoader(testset,batch_size = len(test_labels),collate_fn=collate)
        with torch.no_grad():
            for tbg,tlabel in testset_loader:
                test_pred = best_model(tbg)
                test_pred = torch.max(test_pred,1)[1].tolist()

        ACC = accuracy_score(test_labels,test_pred)
        REC = recall_score(test_labels,test_pred,pos_label = 1, average = 'binary')
        AUC = roc_auc_score(test_labels,test_pred)
        MATRIX = confusion_matrix(test_labels,test_pred,labels = [0,1])
        SP = MATRIX[0,0]*1.0/(MATRIX[0,0]+matrix[0,1])

        model_eval_lines.append([cyp,best_model_index,AUC,REC,SP,ACC])
        
    try:
        os.mkdir('result_csv')
    except Exception:
        print(f'dir <result_csv> be existed')

    loss_graph = pd.DataFrame(epoch_loss_graph,columns = ['cyp','cv','train_loss','valid_loss']).to_csv('epoch_losses/losses.csv',index = False)
    cv_res = pd.DataFrame(cv_res_lines,columns = cv_res_columns).to_csv('result_csv/cv_res.csv',index = False)
    model_eval = pd.DataFrame(model_eval_lines,columns = cv_res_columns).to_csv('result_csv/model_eval.csv',index = False)

    now = time.time()
    print(f'start at {start}, end at {now}, spent {now-start}')

    email = True

    if email: 
        from notification import JobDone
        JobDone(['GCN_workflow'])