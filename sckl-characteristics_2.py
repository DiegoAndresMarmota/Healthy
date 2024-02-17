##1.3.2.6

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer

dataset = pd.DataFrame({
    'IL': [1,2,3,4,5],
    'Mortality': [0.2,0.3,0.4,0.5,0.6]
})

Il = KBinsDiscretizer(
    n_bins=3,
    encode='ordinal',
    strategy='kmeans' 
    )

Il.fit(dataset[['IL']])
Il.transform(dataset[['IL']])

Il_bin = Binarizer(threshold=100)
saving_il = Il_bin.fit_transform(dataset[['IL']])   