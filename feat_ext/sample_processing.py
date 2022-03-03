import wave
import numpy as np
import pandas as pd
from feat_ext.full_feature_extraction import zero_crossing_rate, average_energy, silent_ratio


def load_audio_signal(path, dtype):
    opened_file = wave.open(path, 'rb')
    # Extract Raw Audio from Wav File
    signal = opened_file.readframes(-1)
    signal = np.frombuffer(signal, dtype=dtype)
    signal = signal.astype('int32')
    opened_file.close()
    return signal


def select_30sec_extract(signal, sampling_rate):
    length = signal.size / sampling_rate

    if length > 30:
        middle = int(signal.size / 2)
        lower = int(middle - 15 * sampling_rate)
        higher = int(middle + 15 * sampling_rate)
        return signal[lower:higher]
    else:
        return signal


def get_feature_triplet(signal):
    zcr = zero_crossing_rate(signal)
    avg_eng = average_energy(signal)
    sr = silent_ratio(signal)
    return zcr, avg_eng, sr


'''
Given a pandas dataframe of the extracted feature sets, returns
dataset_train, labels_train, dataset_valid, labels_valid (in this exact order)
'''
def get_normalized_train_valid_sets(data : pd.DataFrame, feature_count = 3):
    # Dataset normalization
    data_mean = data.mean()
    data_std = data.std()

    data_normalized = (data - data_mean) / data_std

    dataset = data_normalized.to_numpy()[:, 1:feature_count + 1]
    labels = data.to_numpy()[:, feature_count + 1].astype(int)

    # Indexes extraction
    N_samples = dataset.shape[0]
    indices = np.random.choice(N_samples, N_samples, replace=False)
    N_train = int(0.8 * N_samples)

    indices_train = indices[:N_train]
    indices_valid = indices[N_train:]

    dataset_train = dataset[indices_train, :]
    dataset_valid = dataset[indices_valid, :]

    labels_train = labels[indices_train]
    labels_valid = labels[indices_valid]

    return dataset_train, labels_train, dataset_valid, labels_valid


'''
Returns the mean and std deviation of the dataset
columns with feature start from left_col_index
'''
def get_feature_mean_and_std(data : pd.DataFrame, features):
    partial_data = data[features]
    # Dataset normalization
    data_mean = partial_data.mean()
    data_std = partial_data.std()
    return data_mean, data_std


'''
Returns a sample normalized with the given vectors of mean and std deviation
'''
def normalized_sample(input, mean, std):
    return (input - mean) / std