#1.3.2.5

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

dataset = pd.DataFrame([
    ("IL-1", "EGF", "GM-CSF", "IFN-A"),
    ("IL-5", "PDGF", "G-CSF", "IFN-B"),
    ("IL-9", "FGF", "M-CSF", "IFN-Y"),
    ("IL-13", "HGF", "CSF", "TNF-A"),
    ("IL-17", "CNTF", "EPO", "TNF-B"),
], columns=[
    "Interleucinas", 
    "Factor de diferenciacion celular", 
    "Factor estimuladores de colonias", 
    "Interferones"])

# print(dataset)

encoder = OneHotEncoder()
encoder.fit(dataset[['Interleucinas']])

codex = encoder.transform(dataset[['Interleucinas']])
new_codex = encoder.transform(dataset[['Interleucinas']]).todense()

# print(codex, new_codex)

new_codex = encoder.inverse_transform(new_codex)

error_encoder = OneHotEncoder(handle_unknown='error', sparse_output=False)
ignore_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

error_encoder.fit(dataset[['Interleucinas']])
ignore_encoder.fit(dataset[['Interleucinas']])

new_data = pd.DataFrame([
    "IL-20"
    ], columns=["Interleucinas"])

ignore_encoder.transform(new_data)

ordinal_encoder = OrdinalEncoder([
    ["Quality 1", "Quality2", "Quality3", "Quality4"]
])
ordinal_encoder.fit(dataset[['Damage']])
ordinal_encoder.transform(dataset[['Damage']])

default_encoder = OrdinalEncoder(categories=[[
    "Quality 1", "Quality2", "Quality3", "Quality4"
]], handle_unknown='use_encoded_value', unknown_value=np.nan)
default_encoder.fit(dataset[['Damage']])