import librosa, librosa.display
import os
import numpy as np

SOUND_PATH = '../Data/genres_original'
NUM_FILES = 500
NUM_FEARTURES = 6
tempo_l, zcr_l, rolloff_l, centroid_l, mel_l, chroma_l, labels_l = [], [], [], [], [], [], []
# tempo_l, zcr_l, rolloff_l, centroid_l, mel_l, chroma_l, labels_l = np.array(tempo_l), np.array(zcr_l), np.array(rolloff_l), np.array(centroid_l), np.array(mel_l), np.array(chroma_l), np.array(labels_l)


def listdirs(rootdir):
    dirs = []
    for it in os.scandir(rootdir):
        if it.is_dir():
            dirs.append(str(it.path))
    return dirs


def get_classes(dirs):
    classes = []
    for dir in dirs:
        pos = dir.rfind('/')
        class_name = dir[pos + 1:]
        classes.append(class_name)
    return classes


def print_2d_arr(l):
    for line in l:
        print(line)


def get_features(filename):
    signal, sr = librosa.load(filename)
    tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=signal, sr=sr)
    pos = filename.rfind('/')
    class_name = filename[pos + 1:-10]
    # zcr, rolloff, centroid, mel, chroma = np.array(zcr), np.array(rolloff), np.array(centroid), np.array(mel), np.array(
    #     chroma)
    print(f'Shape of chroma is {chroma.shape}')
    print(f'Shape of zcr is {zcr.shape}')
    print(f'Shape of rolloff is {rolloff.shape}')
    print(f'Shape of mel is {mel.shape}')
    print(f'Shape of centroid is {centroid.shape}')
    # print_2d_arr(chroma)
    return [tempo, zcr.tolist(), rolloff.tolist(), centroid.tolist(), mel.tolist(), chroma.tolist(), class_name]


def form_dataset(paths):
    for path in paths:
        print(f'---Working on {path}---')
        file_list = list(os.listdir(path))
        for i in range(len(file_list)):
            file_list[i] = path + '/' + file_list[i]
        for filename in file_list[:40]:
            print(f'-Working on {filename}-')
            f = get_features(filename=filename)
            tempo_l.append(f[0])
            zcr_l.append(f[1])
            rolloff_l.append(f[2])
            centroid_l.append(f[3])
            mel_l.append(f[4])
            chroma_l.append(f[5])
            labels_l.append(f[6])
        print(f'Done with {labels_l[-1]} class')
    print('All data processed')


dirs = listdirs(SOUND_PATH)
class_names = get_classes(dirs)
form_dataset(dirs)
