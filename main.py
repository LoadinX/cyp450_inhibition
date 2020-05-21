import os 
import pandas as pd  
import numpy as np 
from itertools import product
#import pubchempy as pcp
from multiprocessing import Pool
from collections import Counter

from standardization import get_standard
from data_sampling import random_under
from calc_descri import calc_fp
from performance_evaluation import performance_function,sp
import DGL_GCN as dl
from sklearn_models import build_sklearn_model,get_performance

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import torch
import time
import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Pool
import matplotlib.pyplot as plt 


global device 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 3 obj need set to(device)
#   model , data , tensor created half way

#reproductbility
np.random.seed(30191375)
torch.manual_seed(30191375)
torch.cuda.manual_seed(30191375)

def local_data(preprocessed,cyp,fpname):
    molecular_description = {
        **{x.upper():'fingerprint' for x in Fingerprints},
        **{'DGL':'dgl_graph'},
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
    elif molecular_description[fpname.upper()] == 'dgl_graph':
        data = local['canonical_smiles']
        label = local.rename(columns={cyp:'labels'})['labels'].apply(lambda x: str(int(x)))
    elif molecular_description[fpname.upper()] == 'descriptor':
        pass
    else:
        print(f'molecular description error, [{fpname}] not supported')
    return data,label

def train_model(data,label,cyp,fpname,method):

    name_str = f'{cyp}_{fpname}_{method}'

    if model_plantform[method.upper()] == 'sklearn':
        t0 = time.time()
        model,best_params,cv_res = build_sklearn_model(data,label,method)
        joblib.dump(model,f'models/{name_str}_model.m')
        print(f'Model {name_str}(cv) | Time(s) {time.time()-t0:.4f}')
        r = [cyp,fpname,method]+get_performance(cv_res).tolist()
        pd.DataFrame([r],columns = store_dict[model_plantform[method.upper()]]).to_csv(f'temp_store/{name_str}_cv.csv',index = False)
    elif model_plantform[method.upper()] == 'dgl_gcn':
        try:
            os.mkdir('epoch_losses')
        except Exception:
            print(f"dir <epoch_losses> be existed")
        train_X,test_X,train_y,test_y = train_test_split(data,label,test_size = 0.2,random_state = 30191375)
        train_graphs,train_labels = dl.df2dataset(pd.concat([train_X,train_y],axis=1))
        #train_graphs,train_labels = dl.df2dataset(pd.concat([data,label],axis=1))
        trainset = dl.dataset(train_graphs,train_labels)

        valid_graphs,valid_labels = dl.df2dataset(pd.concat([test_X,test_y],axis=1))
        validset = dl.dataset(valid_graphs,valid_labels)
        validloader = dl.DataLoader(validset,batch_size=len(valid_labels),shuffle= False,drop_last= False, collate_fn= dl.collate,num_workers=0)
        print('dataset loaded')
        datasetloader = dl.DataLoader(trainset,batch_size=batchsize,shuffle= True,drop_last= False, collate_fn= dl.collate,num_workers=0) #num_workers set to os.cpucount() may cause process hung up
        model = dl.Classifier(in_dim,hidden_dim,trainset.num_classes)
        label = np.array(pd.to_numeric(label,downcast='signed'))
        #pos_weight = np.sum(label)/len(label)
        #loss_func = dl.nn.CrossEntropyLoss(weight = torch.FloatTensor([1-pos_weight,pos_weight]))
        #loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = dl.optim.Adam(model.parameters(),lr = learning_rate,weight_decay=weight_decay)


        if resume:
            checked = []
            for checkpoint in os.listdir('models'):
                if checkpoint.startswith(f'{cyp}_{fpname}_{method}') & checkpoint.endswith(f'_checkpoint.pth.tar'):
                    checked.append(int(checkpoint.split('_',4)[3][5:]))
            stop_at = np.array(checked).max()
            checkpoint_PATH = f'models/{cyp}_{fpname}_{method}_epoch{stop_at}_checkpoint.pth.tar'
            model,optimizer = dl.load_checkpoint(model,optimizer,checkpoint_PATH)
            start_at = stop_at + 1
            print(f'resume training from last checkpoint')
        else:
            start_at = 0
        #[future] add resume for the situation model stopped training, load last checkpoint and continue

        model = model.to(device)
        loss_func = loss_func.to(device)
        

        store = []
        for epoch in range(start_at,EPOCH):
            t0 = time.time()

            model.train()
            epoch_loss = 0
            for iter,(batchgraph,batchlabel) in enumerate(datasetloader):
                batchgraph.set_e_initializer(dl.dgl.init.zero_initializer)
                batchgraph.set_n_initializer(dl.dgl.init.zero_initializer)
                #print(device,batchgraph)
                #batchgraph,batchlabel = batchgraph.to(device),batchlabel.reshape(-1,1).to(device).float()
        
                proba = model(batchgraph)
                #print(proba)
                proba = proba.to(device)
                loss = loss_func(proba,batchlabel)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().item()
            epoch_loss /= (iter+1)

            model.eval()
            for bg,bl in validloader:
                bg.set_e_initializer(dl.dgl.init.zero_initializer)
                bg.set_n_initializer(dl.dgl.init.zero_initializer)

                #bg,bl = bg.to(device),bl.reshape(-1,1).to(device).float()

                proba = model(bg)
                #print('-->',torch.nn.functional.log_softmax(proba,dim=1))
                proba = proba.to(device)
                vloss = loss_func(proba,bl)
                print('PREDICT:',torch.nn.functional.softmax(proba))
                print('LABEL',bl)
                _, indices = torch.max(proba, dim=1)
                correct = torch.sum(indices == bl)
                correct = correct.to(device)
                acc = correct.item() * 1.0 / len(bl)
            

            print(f'Epoch {epoch:05d} | Train Loss {epoch_loss:.4f} | Valid Loss {vloss:.4f}  acc {acc:.4f} | Time(s) {time.time()-t0:.4f}')
            store.append(epoch_loss)
            if (epoch+1)% record_freq == 0:
                torch.save({'epoch': epoch, 
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                            },
                            f'models/{name_str}_epoch{epoch}_checkpoint.pth.tar')
        pd.DataFrame([[cyp,fpname,method,store]],columns = store_dict[model_plantform[method.upper()]]).to_csv(f'epoch_losses/{name_str}_epoch_loss.csv',index = False)
    elif model_plantform[method.upper()] == 'deepchem':
        pass 
    else:
        print(f'model type [{method}] is not supported')
    
    

def test_model(data,label,cyp,fpname,method):

    name_str = f'{cyp}_{fpname}_{method}'
    
    if model_plantform[method.upper()] == 'sklearn':
        model = joblib.load(f'models/{name_str}_model.m')
        proba = model.predict_proba(data)[:,1]       # proba is like [0: prob of 0,1:prob of 1]
        r = performance_function(label,proba)
        pd.DataFrame([[cyp,fpname,method]+r],columns = store_dict[model_plantform[method.upper()]]).to_csv(f'temp_store/{name_str}_out.csv',index = False)
    elif model_plantform[method.upper()] == 'dgl_gcn':
        checkpoint_PATH = f'models/{name_str}_epoch{EPOCH-1}_checkpoint.pth.tar'
        model = dl.Classifier(in_dim,hidden_dim,np.array(pd.to_numeric(label,downcast='signed')).ndim)
        model = dl.load_checkpoint(model,None,checkpoint_PATH)

        acc = dl.evaluate(model,data,label)
        print(acc)
        r = [acc]
        pd.DataFrame([[cyp,fpname,method]+r],columns = store_dict[model_plantform[method.upper()]]).to_csv(f'temp_store/{name_str}_out.csv',index = False)
    elif model_plantform[method.upper()] == 'deepchem':
        pass 
    

def merge_file(keyword,path):
    merged = pd.DataFrame()
    for filename in os.listdir(path):
        df = pd.read_csv(f'{path}{filename}')
        if keyword in filename:
            merged = pd.concat([merged,df])
    merged.to_csv(f'{keyword}_merged.csv',index =False)


def consensus(data,label,tocombine):
    for cyp,fpname,method in tocombine:
        test_model(data,label,cyp,fpname,method)


# opt_param
model_plantform = {
        **{x:'sklearn' for x in ['SVM', 'NN', 'RF', 'KNN','GBDT','XGBOOST']},
        **{y:'dgl_gcn' for y in ['GCN']},
        **{z:'deepchem' for z in ['GCNN']}
    }

store_dict = { 
        'sklearn':['endpoint','fp','method','AUC','SE','SP','ACC'],
        'dgl_gcn':['endpoint','fp','method','epoch_loss'],
        'deepchem':['endpoint','fp','method']
    }  

global EPOCH,record_freq,rus,resume,in_dim,hidden_dim,batchsize,learning_rate,weight_decay
EPOCH = 1000
record_freq = 50
rus = False
resume = False
in_dim = 1 # dgl_graph 
hidden_dim = 1024
batchsize = 128
learning_rate = 1e-1
weight_decay = 1e-3



if __name__ == "__main__":

    Fingerprints = ['MACCS','RDKFP','MorganFP','AvalonFP']
    Methods = ['svm', 'nn', 'rf', 'knn']

    start = time.time()

    origin = pd.read_csv(f'material/extracted_1851.csv')
    print('processing compounds......')
    # match and standardise
    if 'standard_compounds.csv' in os.listdir('material/'):
        result = pd.read_csv('material/standard_compounds.csv')
    else:
        result = get_standard()

    df = origin.dropna(axis = 0,subset = ['CID'])
    df['cid'] = df['CID']
    df['canonical_smiles'] = None
    df.update(result[['canonical_smiles','cid']])

    #local model for each CYP450 enzyme
    cyps = []
    for col in df.columns:
        if 'cyp' in col:
            cyps.append(col) 
    mix = df[cyps+['canonical_smiles']]
    #mix = df[cyps+['canonical_smiles']].iloc[:100] ###### demo mix 

    try:
        os.mkdir('models')
    except Exception:
        print(f"dir <models> be existed")

    try:
        os.mkdir('temp_store')
    except Exception:
        print(f"dir <temp_store> be existed")

    ## compare balanced
    #need_balance = [['cyp1a2','RDKFP','svm'],['cyp2c19','RDKFP','svm'],['cyp2c9','MACCS','rf'],['cyp2d6','MorganFP','svm'],['cyp3a4','MACCS','svm']]

    sklearns = [x for x in product(cyps,Fingerprints,Methods)]
    deeplearnings = [x for x in product(cyps,['dgl'],['gcn'])]
    demo = [['cyp1a2','dgl','gcn']]

    target = demo
    #target = sklearns + deeplearnings
    #target = sklearns
    #target = deeplearnings
    
    if resume:
        drop = []
        for exist in os.listdir('models'):
            if 'checkpoint' in exist:
                cyp,fpname,method,e,others = exist.split('_',4)
                if int(e[5:]) == EPOCH:
                    drop.append((cyp,fpname,method))
            else:
                cyp,fpname,method,others = exist.split('_',3)
                drop.append((cyp,fpname,method))
        targets = [x for x in target if x not in drop]
    else:
        targets = [x for x in target]

    
    for cyp,fpname,method in target: # target has removed done-model if resume
        print(f'------------------{cyp}_{fpname}_{method}------------------')
        local = mix[[cyp]+['canonical_smiles']].dropna()
        local = local.drop_duplicates('canonical_smiles',keep = 'first')
        print(f'origin with labels : {Counter(local[cyp])}')
        data,label = local_data(local,cyp,fpname)

        # data sampling strategy
        if rus:
            data,label = random_under(data,label)
            print(f'RandomUnderSampled : {Counter(label)}')
        else:
            print(f'DataLabel Detail : {Counter(label)}')

        train_model(data,label,cyp,fpname,method)
        '''
        X_train,X_test,y_train,y_test = train_test_split(data,label,test_size = 0.2,random_state = 30191375)

        train_model(X_train,y_train,cyp,fpname,method)
        test_model(X_test,y_test,cyp,fpname,method) 
        '''

    now = time.time()
    print(f'model construction spent {now-start:.4f}')

    for keyword in ['cv']:
        merge_file(keyword,'temp_store/')

    email = False


    if email: 
        import datetime
        from notification import JobDone
        JobDone([f'{datetime.data.today()}'])



    '''
    [future]
    def cid2detail(cid):
        detail = pcp.get_compounds(int(cid),'cid',as_dataframe = True).reset_index(drop = False)
        return detail

    def url_request_PUG(cids):
        # cids must be int seperated by dots
        properties = 'MolecularFormula,MolecularWeight,CanonicalSMILES'
        cid_request = ''
        for cid in cids:
            cid_request += f'{int(cid)},'
        cid_request = cid_request[:-1]
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_request}/property/{properties}/CSV'
        get = urllib.request.urlopen(url)
        get.read()
    '''

    '''
    # error PUG.ServerBusy 
    with Pool(processes= os.cpu_count()) as pool:
        for d in pool.map(cid2detail,df['CID']):
            result = pd.concat([result,d])    
    '''

    '''
    #%%
    from rdkit import Chem
    import pandas as pd

    mols_supply = Chem.SDMolSupplier('CID.sdf')
    mols_supply[0].GetAtoms()

    #%%
    l = [1,1,1,1,1,1,1]
    l[[1,2,4]]


    #%%
    import urllib.request
    import pandas as pd 
    df = pd.read_csv('extracted_1851.csv')
    cids = df['CID'].dropna()[0:1000]
    # cids must be int seperated by dots
    properties = 'MolecularFormula,MolecularWeight,CanonicalSMILES'
    cid_request = ''
    for cid in cids:
        cid_request += f'{int(cid)},'
    cid_request = cid_request[:-1]
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_request}/property/{properties}/CSV'
    get = urllib.request.urlopen(url)

    #%%
    import numpy as np 
    import pandas as pd 
    df = pd.DataFrame([[1,2,np.nan,4]])
    df.dropna(axis = 1)
    '''
