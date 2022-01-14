import numpy as np
import pandas as pd
import os
import wave


def zero_crossing_rate(signal):
    sign = np.sign(signal)
    ZCR=0
    for i in range(1, sign.size):
        ZCR += np.absolute(sign[i] - sign[i-1])
    return ZCR/(2*sign.size)


def average_energy(signal):
    return np.mean(signal ** 2)


def silent_ratio(signal):
    threshold = np.sqrt(average_energy(signal)) * 0.8
    sr = 0
    for i in range(signal.size):
        if signal[i] < threshold:
            sr += 1

    return sr / signal.size


def normalize_dataset(dataset_path):
    set = pd.read_csv(dataset_path)
    set_mean = set.mean()
    set_std = set.std()
    rdata_normalized = (set - set_mean) / set_std
    out_path = dataset_path[:-4] + '_normalized.csv'
    rdata_normalized.to_csv(out_path)


def create_dataset():
    featurelist = list()
    for root, dir, files in os.walk("./genres/"):
        for k, genre in enumerate(sorted(dir)):
            print('Started genre {} ({})'.format(k, genre))
            for root2, dir2, tracks in os.walk("./genres/" + genre + '/'):
                for track in sorted(tracks):
                    wav_file = wave.open(root2 + track, 'rb')
                    signal = wav_file.readframes(-1)
                    signal = np.frombuffer(signal, dtype='int16')
                    signal = signal.astype('int32')
                    wav_file.close()
                    featurelist.append(zero_crossing_rate(signal))
                    featurelist.append(average_energy(signal))
                    featurelist.append(silent_ratio(signal))
                    featurelist.append(k)
                    print('Working on track {}'.format(track))
            print('Ended genre {} ({})'.format(k, genre))
    featurearr = np.array(featurelist).reshape(len(featurelist)//4, 4)
    df = pd.DataFrame(featurearr, columns=['ZCR', 'AVERAGE_ENERGY', 'SILENT_RATIO', 'CLASS'])
    df.to_csv('test_dataset.csv')

