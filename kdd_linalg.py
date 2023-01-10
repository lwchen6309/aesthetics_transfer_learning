import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


load_dct = True
if load_dct:
    dct = np.load('KDD.npz')
    w = dct['w']
    v = dct['v']
    kdd = dct['kdd']
    # ktd = dct['ktd']
    # kdd_re = np.dot(v, w[:,np.newaxis] * v.T)
    
    vit_feature_file = 'cifar_vit.npz'
    vit_dct = np.load(vit_feature_file)
    x_train_full = vit_dct['x_train']
    y_train_full = vit_dct['y_train']
    x_test = vit_dct['x_test']
    y_test = vit_dct['y_test']

    x_train_full = x_train_full[:20000]
    y_train_full = y_train_full[:20000]


n = 5
x = kdd[:,:n][:n,:]
w, v = np.linalg.eigh(x)
# re_k = np.dot(v, w[:,np.newaxis] * v.T)

cut_off = 2
h0 = np.array(x)
h0[:,cut_off:] = 0
h0[cut_off:,:] = 0
h0[np.arange(n),np.arange(n)] = np.diag(x)

h1 = np.array(x)
h1[:,:cut_off][:cut_off,:] = 0
h1[np.arange(n),np.arange(n)] = 0

np.testing.assert_allclose(x, h0+h1)

w0, v0 = np.linalg.eigh(h0)
h0_inv = np.dot(v0, 1/w0[:,np.newaxis] * v0.T)

metric = np.linalg.norm(h1@h0_inv, axis=1)

raise Exception
# Reduce the dimension of the data to 2 dimensions using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(x_train_full)
X_reduced

trunc_w = w[-100:]
trunc_v = v[:,-100:]
# kdd_re = np.dot(trunc_v, trunc_w[:,np.newaxis] * trunc_v.T)

dim_norm = np.abs(trunc_v).sum(axis=1)

# mu = ktd.T@np.linalg.inv(kdd)@y_train_full
# accuracy = np.argmax(mu, axis=1) == np.argmax(y_test, axis=1)

