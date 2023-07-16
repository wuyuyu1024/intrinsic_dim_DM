import os
import sys
sys.path.append('../sdbm/code')
sys.path.append('../DeepView/deepview')
sys.path.append('../dbm_evaluation')
# import make blobs
from sklearn.datasets import make_blobs
# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import torch as T
import torch.linalg as LA

from ssnp import SSNP
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

from map_evaluation import P_wrapper, MapBuilder

sys.path.append('../GAN_inverse_projection')
from utils import GANinv, CGANinv
from umap import UMAP
from LID import ID_finder_T

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]) # Notice here
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


data_dir = '../sdbm/data'
data_dirs = [
    # 'blobs_dim5_n1000',
    # 'blobs_dim50_n1000',
    # 'blobs_dim100_n1000',
    # 'har', 
    'mnist', 
    # 'fashionmnist', 
    # 'reuters', 
    ]
datasets_real = {}

for d in data_dirs:
    dataset_name = d
    if not 'blob' in d:
        X = np.load(os.path.join(data_dir, d,  'X.npy'))
        y = np.load(os.path.join(data_dir, d, 'y.npy'))

    #blobs
    else:
        dim = int(d.split('_')[1][3:])
        print(dim)
        X, y = make_blobs(n_samples=1500, n_features=dim, centers=5, cluster_std=1, random_state=0)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    ######
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))

    if 'blob' in d:
        train_size = 1000
    else:
        train_size = min(int(n_samples*0.9), 2000)
    test_size = 5000 # inverse
    
    dataset =\
        train_test_split(X, y, train_size=train_size, random_state=420, stratify=y)
    datasets_real[dataset_name] = dataset
    # print(X.shape)

    ## clip dataset[1] and dataset[3] to test_size if they are larger
    if dataset[1].shape[0] > test_size:
        dataset[1] = dataset[1][:test_size]
        dataset[3] = dataset[3][:test_size]



projectors = {
            # 'DBM_orig_torch': P_wrapper(NNinv_Torch=1),
            # 'DeepView_0.65': P_wrapper(deepview=1),
            # 'DBM_orig_keras': P_wrapper(NNinv_Keras=1),
            'SSNP' : P_wrapper(ssnp=1),
            }



###########################
#SAVE DIR
save_dir = './LID_results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



for data_name, dataset in datasets_real.items():
    print(data_name)
    print('='*20)
    print(f"processing {data_name}")
    X_train, X_test, y_train, y_test = dataset
    n_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    clf = LogisticRegression(max_iter=1000, random_state=420)
    clf.fit(X_train, y_train)
    print(f"training accuracy: {clf.score(X_train, y_train)}")
    print(f"test accuracy: {clf.score(X_test, y_test)}")


    for proj_name, proj in projectors.items():
        print(f"processing {data_name} with {proj_name}")

        proj.fit(X_train, y_train, clf)
        if proj.ssnp:
            X_train_2d = proj.transform(X_train)
        else:
            X_train_2d = proj.P.embedding_

        map_builder = MapBuilder(clf, proj, X_train, y_train, grid=200)
        alpha, labels = map_builder.get_prob_map()
        GM = map_builder.get_gradient_map()
        if 'blob' in data_name:
            sampling_size = input_dim + 10
        else:
            sampling_size = 30
        LID_finder = ID_finder_T(X_train_2d, proj, grid=100, sample_size=sampling_size, device='cpu')
        LID_evalues = LID_finder.LID_eval.to('cpu').numpy()

        ### save LID_evalues, (alpha, labels), GM, X_train_2d, y_train separately
        save_path = os.path.join(save_dir, f"{data_name}_{proj_name}.npz")
        np.savez(save_path, LID_evalues=LID_evalues, alpha=alpha, labels=labels, GM=GM, X_train_2d=X_train_2d, y_train=y_train)
        print(f"saved to {save_path}")
        




            

