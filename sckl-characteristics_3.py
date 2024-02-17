##1.3.2.7

import pandas as pd 

from sklearn.impute import SimpleImputer

dataset = pd.DataFrame([
    ('A', 1, 1.2, -1.1, None),
    ('B', 2, 2.2, -2.1),
    ('C', 3, 3.2, -3.1),
    ('D', 4, 4.2, -4.1, None),
    ('E', 5, 5.2, -5.1),
], columns=[
    'cepa', 'id', 'disposicion', 'valor', 'mutabilidad'
])


# The code is using the `SimpleImputer` class from the `sklearn.impute` module to fill missing values
# in the 'valor' column of the dataset.
cepa = SimpleImputer(strategy='constant', fill_value='Cepa')
cepa.fit(dataset[['valor']])
dataset['valor_asignado'] = cepa.transform(dataset[['valor']])


# The code is using the `SimpleImputer` class from the `sklearn.impute` module to fill missing values
# in the 'disposicion' column of the dataset.
tipo_de_cepa = SimpleImputer(strategy='median')
tipo_de_cepa.fit(dataset[['disposicion']])
dataset['disposicion_asignado'] = tipo_de_cepa.transform(dataset[['disposicion']])


# The code is using the `SimpleImputer` class from the `sklearn.impute` module to fill missing values
# in the 'mutabilidad' column of the dataset.
mutabilidad = SimpleImputer(strategy='most_frequent', missing_values=pd.NA)
mutabilidad.fit(dataset[['mutabilidad']])
dataset['mutabilidad_asignado'] = mutabilidad.transform(dataset[['mutabilidad']])
mutabilidad.transform(dataset[['mutabilidad']]).squeeze()


# The code is using the `SimpleImputer` class from the `sklearn.impute` module to fill missing values
# in the 'id' column of the dataset.
constant_id = SimpleImputer(strategy='constant', fill_value=10)
constant_id.fit(dataset[['id']])
dataset['id_asignado'] = constant_id.transform(dataset[['id']])
constant_id.transform(dataset[['id']]).squeeze()