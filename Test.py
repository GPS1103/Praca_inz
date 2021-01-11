import pickle
import MFCC
import os
import numpy as np
import sys
from scipy.io import wavfile
from python_speech_features import mfcc

modelpath = "models/"
gmm_files = gmm_files = [os.path.join(modelpath, fname) for fname in
                         os.listdir(modelpath) if fname.endswith('.gmm')]
models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
ubm = pickle.load(open('testing_samples/UBM.gmm', 'rb'))
speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname
            in gmm_files]
sample_rate, data = wavfile.read('testing_samples/' + sys.argv[1])
vector = mfcc(data, sample_rate, nfft=2048)
log_likelihood = np.zeros(len(models))
ubmscores = np.array(ubm.score(vector))
print(ubmscores.sum())
for i in range(len(models)):
    gmm = models[i]  # checking with each model one by one
    print(speakers[i])
    scores = np.array(gmm.score(vector))
    log_likelihood[i] = scores.sum()/ubmscores.sum()
    print(log_likelihood[i])
