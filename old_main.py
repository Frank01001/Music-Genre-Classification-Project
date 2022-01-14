import wave
import matplotlib.pyplot as plt
import numpy as np
import feature_extraction

# feature_extraction.create_dataset()

test_file = wave.open('genres/blues/blues.00000.wav', 'rb')
# Extract Raw Audio from Wav File
signal = test_file.readframes(-1)
signal = np.frombuffer(signal, dtype='int16')
test_file.close()

# print(signal.shape)

print(feature_extraction.average_energy(signal))

# plt.figure(1)
# plt.title("Signal Wave...")
# plt.plot(signal)
# plt.show()

