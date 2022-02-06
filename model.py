import pandas as pd
import numpy as np
import pickle
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Reshape, Dropout, SimpleRNN
from keras.layers.merge import Concatenate
from keras.models import Model
import os
import tensorflow as tf
import keras as k
from preprocessing import get_features

genres = ['blues', 'classical', 'country', 'disco', 'jazz', 'hiphop', 'pop', 'rock', 'reggae', 'metal']

def train_test_split(l):
    train_l, test_l = [], []
    for i in range(1, 201):
        if i != 0 and i % 19 == 0 or i % 20 == 0:
            test_l.append(l[i - 1])
        else:
            train_l.append(l[i - 1])
    return train_l, test_l


dataset = pickle.load(open('dataset1.pickle', 'rb'))
labels = dataset['labels']
tempo = dataset['tempo']
centroid = dataset['centroid']
rolloff = dataset['rolloff']
mel = dataset['mel']
zcr = dataset['zcr']
chroma = dataset['chroma']
tempo_sample = tempo[0]
centroid_sample = np.array(centroid[0])
rolloff_sample = np.array(rolloff[0])
mel_sample = np.array(mel[0])
zcr_sample = np.array(zcr[0])
chroma_sample = np.array(chroma[0])

# train_labels, test_labels = train_test_split(labels)
train_tempo, test_tempo = train_test_split(tempo)
train_centroid, test_centroid = train_test_split(centroid)
train_rolloff, test_rolloff = train_test_split(rolloff)
train_mel, test_mel = train_test_split(mel)
train_zcr, test_zcr = train_test_split(zcr)
train_chroma, test_chroma = train_test_split(chroma)

train_labels_num = []
# print(train_labels)
for i in range(10):
    for _ in range(18):
        train_labels_num.append(i)

train_labels_num = np.array(train_labels_num)
print(f'train_tempo: {train_tempo}')

print(f"Shape of centroid is {centroid_sample.shape}")
print(f"Shape of rolloff is {rolloff_sample.shape}")
print(f"Shape of mel is {mel_sample.shape}")
print(f"Shape of zcr is {zcr_sample.shape}")
print(f"Shape of chroma is {chroma_sample.shape}")

train_tempo = np.array(train_tempo)
train_centroid = np.array(train_centroid)
train_rolloff = np.array(train_rolloff)
train_mel = np.array(train_mel)
train_zcr = np.array(train_zcr)
train_chroma = np.array(train_chroma)
print(f'train_tempo shape is: {train_tempo.shape}')
print(f"train_centroid shape is: {train_centroid.shape}")
print(f"train_rolloff shape is: {train_rolloff.shape}")
print(f"train_mel shape is: {train_mel.shape}")
print(f"train_zcr shape is: {train_zcr.shape}")
print(f"train_chroma shape is: {train_chroma.shape}")

idx = 1
test_tempo = np.array(test_tempo[:idx])
test_centroid = np.array(test_centroid[:idx])
test_rolloff = np.array(test_rolloff[:idx])
test_mel = np.array(test_mel[:idx])
test_zcr = np.array(test_zcr[:idx])
test_chroma = np.array(test_chroma[:idx])


def baseline_model():
    chroma_layer = Input(shape=(12, 1280, 1))
    mel_layer = Input(shape=(128, 1280, 1))
    rolloff_layer = Input(shape=(1280,))
    centroid_layer = Input(shape=(1280,))
    zcr_layer = Input(shape=(1280,))
    tempo_layer = Input(shape=(1,))

    x = Conv2D(filters=4, input_shape=(12, 1280, 1), kernel_size=(3, 3), activation='relu')(chroma_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Reshape((12780,))(x)
    x = Dense(100, activation='relu')(x)

    y = Conv2D(filters=4, input_shape=(128, 1280, 1), kernel_size=(3, 3), activation='relu')(mel_layer)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Reshape((161028,))(y)
    y = Dense(100, activation='relu')(y)

    z = Dense(1280, activation='relu')(centroid_layer)
    z = Dense(100, activation='relu')(z)
    z = Dense(20, activation='relu')(z)

    a = Dense(1280, activation='relu')(rolloff_layer)
    a = Dense(100, activation='relu')(a)
    a = Dense(20, activation='relu')(a)

    b = Dense(1280, activation='relu')(zcr_layer)
    b = Dense(100, activation='relu')(b)
    b = Dense(20, activation='relu')(b)

    c = Dense(1, activation='relu')(tempo_layer)
    
    combined = Concatenate()([x, y, z, a, b, c]) 

    # final layer
    final = Dense(100, activation='relu')(combined)
    final = Dense(20, activation='relu')(final)
    final = Dense(1)(final)

    model = Model(inputs=[chroma_layer, mel_layer, centroid_layer, rolloff_layer, zcr_layer, tempo_layer],
                  outputs=final)  # add tempo here
    model.compile(loss='mse', optimizer=tf.optimizers.Adam(), metrics=['mae'])
    return model


model = baseline_model()
# model.fit([train_chroma, train_mel, train_centroid, train_rolloff, train_zcr, train_tempo], train_labels_num, steps_per_epoch=10, epochs=5, validation_split = 0.2, validation_steps=1, verbose=True)
model.fit([train_chroma, train_mel, train_centroid, train_rolloff, train_zcr, train_tempo], train_labels_num, epochs=5,
          verbose=True)
# print(int(abs(model.predict([test_chroma, test_mel, test_centroid, test_rolloff, test_zcr, test_tempo])/20)))
print(int(model.predict([test_chroma, test_mel, test_centroid, test_rolloff, test_zcr, test_tempo])))

def get_genre_from_wav(path):
    features = get_features(path)
    features = features[:-1]
    idx = int(model.predict(features))
    return genres[idx]
