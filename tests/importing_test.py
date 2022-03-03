from feat_ext.sample_processing import *
from nearest_centroid.nearest_centroid_classifier import NearestCentroidClassifier

genre_names = ['blues', 'classical', 'country', 'disco', 'hipop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

path_to_wav = ''
path_to_extracted_dataset = ''
signal = load_audio_signal(path_to_wav, 'int32')
signal = select_30sec_extract(signal, sampling_rate=44100)

zcr, avg_eng, sr = get_feature_triplet(signal)
sample = np.array([zcr, avg_eng, sr])

data = pd.read_csv(path_to_extracted_dataset)

training_set, training_labels, valid_set, valid_labels = get_normalized_train_valid_sets(data)

ncc = NearestCentroidClassifier()
ncc.train(training_set, training_labels)

mean, std = get_feature_mean_and_std(data, features=['ZCR', 'AVERAGE_ENERGY', 'SILENT_RATIO'])
sample_normalized = normalized_sample(sample, mean, std)
predicted = ncc.classify(sample_normalized, genres_to_classify=genre_names)

print('Classified %s as %s' % (path_to_wav, predicted))

