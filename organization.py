import preprocessing
import pandas as pd
import numpy as np
import pickle

LIST_SIZE = 1280


def separation(arr):
    l = []
    for items in arr:
        l.append(items)
    return l


def size_standardize_1d(list_1d):
    l_standardized = []
    for i in range(LIST_SIZE):
        l_standardized.append(list_1d[i])
    return l_standardized


def size_standardize_2d(list_2d):
    l_standardized = []
    for item in list_2d:
        l_standardized.append(item[:1280])
    return l_standardized


tempo = preprocessing.tempo_l
chroma = preprocessing.chroma_l
centroid = preprocessing.zcr_l
rolloff = preprocessing.rolloff_l
mel = preprocessing.mel_l
zcr = preprocessing.zcr_l
labels = preprocessing.labels_l
print(type(mel))

# chroma = separation(chroma)
# centroid = separation(centroid)
# rolloff = separation(rolloff)
# mel = separation(mel)
# zcr = separation(zcr)

data = []

for i in range(200):
    data.append(
        [labels[i], tempo[i], size_standardize_2d(chroma[i]), size_standardize_1d(rolloff[i][0]), size_standardize_1d(centroid[i][0]),
         size_standardize_2d(mel[i]), size_standardize_1d(zcr[i][0])])
    print('---size verification---')
    # print(np.array(size_standardize_2d(chroma[i])).shape)
    # print(np.array(size_standardize_1d(rolloff[i][0])).shape)
df = pd.DataFrame(data, columns=['labels', 'tempo', 'chroma', 'rolloff', 'centroid', 'mel', 'zcr'])
print(df.head(10))
pickle.dump(df, open("dataset1.pickle", 'wb'))
dataset = pickle.load(open("dataset1.pickle", 'rb'))
