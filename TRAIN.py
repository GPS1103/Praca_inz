import MFCC
import pickle
import os
import numpy as np
import sys
from sklearn.mixture import GaussianMixture

samples = os.listdir('training_samples/' + sys.argv[1] + '/')

features = np.asarray(())
for sample in samples:
    mfcc_data = MFCC.main('training_samples/' + sys.argv[1] + '/' + sample)
    if features.size == 0:
        features = mfcc_data
    else:
        features = np.vstack((features, mfcc_data))
gmm = GaussianMixture(n_components=len(samples),
                      max_iter=200, covariance_type='diag', n_init=3)
gmm.fit(features)
savefile = sys.argv[1] + '.gmm'
pickle.dump(gmm, open('models/' + savefile, 'wb'))
print('Modeling for ' + sys.argv[1] + ' completed')
