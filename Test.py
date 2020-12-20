import pickle
import MFCC
import os
import numpy as np
import sys
modelpath = "models/"
gmm_files = gmm_files = [os.path.join(modelpath, fname) for fname in
                         os.listdir(modelpath) if fname.endswith('.gmm')]
models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname
            in gmm_files]
vector = MFCC.main('testing_samples/' + sys.argv[1])
log_likelihood = np.zeros(len(models))

for i in range(len(models)):
    gmm = models[i]  # checking with each model one by one
    print(speakers[i])
    scores = np.array(gmm.score(vector))
    log_likelihood[i] = scores.sum()
    print(log_likelihood[i])
