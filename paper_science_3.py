from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher

data = [
    {'exposicion_a': 'presente', 'casos': 1021, 'fallecidos': 191, 'expuestos': 1931},
    {'exposicion_b': 'pasado', 'casos': 1141, 'fallecidos': 89, 'expuestos': 2788},
    {'exposicion_c': 'pasado', 'casos': 1574, 'fallecidos': 101, 'expuestos': 4798},
    {'exposicion_d': 'pasado', 'casos': 1358, 'fallecidos': 175, 'expuestos': 4605},
]

# The code is using the `DictVectorizer` class from the `sklearn.feature_extraction` module to convert
# a list of dictionaries (`data.morbility`) into a numerical representation that can be used for machine
# learning algorithms.
vectorizer = DictVectorizer(sparse=False)
vectorizer.fit(data)
DictVectorizer(sparse=False)
vectorizer_data = vectorizer.transform(data)

print(vectorizer_data)

# [[1.021e+03 1.000e+00 0.000e+00 0.000e+00 0.000e+00 1.931e+03 1.910e+02]
#  [1.141e+03 0.000e+00 1.000e+00 0.000e+00 0.000e+00 2.788e+03 8.900e+01]
#  [1.574e+03 0.000e+00 0.000e+00 1.000e+00 0.000e+00 4.798e+03 1.010e+02]
#  [1.358e+03 0.000e+00 0.000e+00 0.000e+00 1.000e+00 4.605e+03 1.750e+02]]

print(vectorizer.feature_names_, vectorizer.vocabulary_)

# [
    # 'casos', 'exposicion_a=presente', 'exposicion_b=pasado', 'exposicion_c=pasado', 'exposicion_d=pasado', 'expuestos', 'fallecidos'
# ] 

# {'casos': 0, 
# 'exposicion_a=presente': 1, 
# 'exposicion_b=pasado': 2, 
# 'exposicion_c=pasado': 3, 
# 'exposicion_d=pasado': 4, 
# 'expuestos': 5, 
# 'fallecidos': 6
# }

# The code is using the `FeatureHasher` class from the `sklearn.feature_extraction` module to convert
# the list of dictionaries (`data`) into a numerical representation using feature hashing.
hasher_one = FeatureHasher(n_features=10)
hasher_analysis = hasher_one.fit_transform(data)
print(hasher_analysis.todense())

# [
#     [ 0.000e+00  0.000e+00  0.000e+00  0.000e+00  1.021e+03 -1.000e+00
#     0.000e+00  0.000e+00  2.122e+03  0.000e+00]
#     [ 0.000e+00  0.000e+00  0.000e+00  0.000e+00  1.141e+03  0.000e+00
#     0.000e+00  1.000e+00  2.877e+03  0.000e+00]
#     [ 0.000e+00 -1.000e+00  0.000e+00  0.000e+00  1.574e+03  0.000e+00
#     0.000e+00  0.000e+00  4.899e+03  0.000e+00]
#     [ 0.000e+00  0.000e+00  0.000e+00  0.000e+00  1.358e+03  0.000e+00
#     0.000e+00  0.000e+00  4.779e+03  0.000e+00]
# ]

# The code is creating an instance of the `FeatureHasher` class with `n_features=10` and
# `input_type='string'`. Then, it is using this instance to transform the `data` using the `transform`
# method. Finally, it is printing the dense representation of the transformed data using the `todense`
# method.
hasher_one = FeatureHasher(n_features=10, input_type='string')
hasher_analysis = hasher_one.transform(data)
print(hasher_analysis.todense())


# [
#     [ 0.  0.  0.  0.  1.  0.  0. -1.  2.  0.]
#     [ 0.  0.  0.  0.  1.  0.  0. -1.  2.  0.]
#     [ 0.  0.  0. -1.  1.  0.  0.  0.  2.  0.]
#     [-1.  0.  0.  0.  1.  0.  0.  0.  2.  0.]
# ]
