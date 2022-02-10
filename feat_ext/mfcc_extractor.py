import librosa
import numpy as np


def get_mfcc_feature_set(signal, sampling_rate, n_coeffs = 13):
    signal = signal.astype('float64')
    mfccs = librosa.feature.mfcc(signal, sr=sampling_rate, n_mfcc=n_coeffs)

    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

    num_windows = mfccs.shape[1]
    random_sample_index = np.random.choice(num_windows, 1)[0]

    feature_set = list()
    feature_set.extend(mfccs[:, random_sample_index].T.tolist())
    feature_set.extend(mfccs_delta[:, random_sample_index].T.tolist())
    feature_set.extend(mfccs_delta2[:, random_sample_index].T.tolist())
    return np.array(feature_set)
