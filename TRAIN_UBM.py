import MFCC
import pickle
import os
import numpy as np
import sys
from sklearn.mixture import GaussianMixture
from scipy.io import wavfile
from python_speech_features import mfcc
dirs = os.listdir('training_samples')

features = np.asarray(())
for dir in dirs:
    samples = os.listdir('training_samples/' + dir + '/')
    for sample in samples:
        sample_rate, data = wavfile.read(
            'training_samples/' + dir + '/' + sample)
        mfcc_data = mfcc(data, sample_rate, nfft=2048)
        if features.size == 0:
            features = mfcc_data
        else:
            features = np.vstack((features, mfcc_data))
gmm = GaussianMixture(n_components=len(samples),
                      max_iter=200, covariance_type='diag', n_init=3)
gmm.fit(features)
savefile = 'UBM.gmm'
pickle.dump(gmm, open('testing_samples/' + savefile, 'wb'))
print('Modeling for UBM completed')
