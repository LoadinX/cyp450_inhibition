# data sampling methods 
## can be under sampling, over sampling


import pandas as pd 
from collections import Counter
import numpy as np


def random_under(data,label):
    from imblearn.under_sampling import RandomUnderSampler
    assert isinstance(label,pd.Series)
    before = Counter(label)
    rus = RandomUnderSampler(random_state=0)
    x,y = rus.fit_resample(data,label)
    after = Counter(y)
    print(f'{before} --> {after}')
    return np.array(x),np.array(y)