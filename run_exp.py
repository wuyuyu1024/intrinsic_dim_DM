import os
import sys
sys.path.append('../sdbm/code')
sys.path.append('../DeepView/deepview')
sys.path.append('../dbm_evaluation')
sys.path.append('../InverseProjections')
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

from lamp import Pinv_ilamp
# from NNinv import NNinv_torch
from rbf_inv import RBFinv

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
# import skdim

from map_evaluation import P_wrapper, MapBuilder
from ssnp import SSNP

sys.path.append('../GAN_inverse_projection')
# from utils import GANinv, CGANinv
from umap import UMAP
from LID import ID_finder_T, get_data_LID, get_eigen_general , ID_finder_np
from gradient_map import get_gradient_map

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


class Simple_P_wrapper:
    def __init__(self, P, Pinv, bottleneck_dim=2):
        self.P = P
        self.Pinv = Pinv
        self.bottleneck_dim = bottleneck_dim
    def __call__(self, x):
        return self.P(x)
    def transform(self, x):
        return self.P.transform(x)
    def inverse_transform(self, x):
        return self.Pinv.transform(x)

    def fit(self, x, y=None, clf=None):
        # self.P.fit(x)
        self.X2d = self.P.fit_transform(x)
        self.Pinv.fit(self.X2d, x )
        return self


data_dir = '../sdbm/data'
data_dirs = [
    # 'blobs_dim3_n5000_y5',
    # 'blobs_dim10_n5000_y5',
    # 'blobs_dim30_n5000_y5',
    # 'blobs_dim100_n5000_y5',
    # 'blobs_dim300_n5000_y5',

    # 'blobs_dim3_n5000_y2',
    # 'blobs_dim10_n5000_y2',
    # 'blobs_dim30_n5000_y2',
    # 'blobs_dim100_n5000_y2',
    # 'blobs_dim300_n5000_y2',

    # 'blobs_dim3_n5000_y3',
    # 'blobs_dim10_n5000_y3',
    # 'blobs_dim30_n5000_y3',
    # 'blobs_dim100_n5000_y3',
    # 'blobs_dim300_n5000_y3',
    # 'blobs_dim3_n500_y10',
    # 'blobs_dim3_n5000_y10',
    # 'blobs_dim10_n5000_y10',
    
    # 'blobs_dim30_n5000_y10',
    'blobs_dim60_n5000_y10',
    # 'blobs_dim100_n5000_y10',

    # # 'blobs_dim300_n1500_y10',

    # 'har', 
    # 'mnist', 
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
        n_class = int(d.split('_')[3][1:])
        X, y = make_blobs(n_samples=6000, n_features=dim, centers=n_class, cluster_std=1, random_state=0)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    ######
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))

    if 'blob' in d:
        train_size = dim = int(d.split('_')[2][1:])
    else:
        train_size = min(int(n_samples*0.9), 5000)
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
            'SDBM' : P_wrapper(ssnp=1),
            'DBM': P_wrapper(NNinv_Torch=1, ),
            'DeepView': P_wrapper(deepview=1),
            'UMAP+iLAMP': Simple_P_wrapper(UMAP(random_state=42), Pinv_ilamp()),
            'UMAP+RBF': Simple_P_wrapper(UMAP(random_state=42), RBFinv()),
            # 'DBM_orig_keras': P_wrapper(NNinv_Keras=1),
            
            # 'SSNP_3' : SSNP(patience=5, opt='adam', bottleneck_activation='linear', verbose=0, bottleneck_dim=3),
            # 'SSNP_10' : SSNP(patience=5, opt='adam', bottleneck_activation='linear', verbose=0, bottleneck_dim=10),
            # 'SSNP_30' : SSNP(patience=5, opt='adam', bottleneck_activation='linear', verbose=0, bottleneck_dim=30),
            
            # 'DBM_2' : P_wrapper(NNinv_Torch=1, bottleneck_dim=2),
            # 'DBM_5' : P_wrapper(NNinv_Torch=1, bottleneck_dim=5),
            # 'DBM_10' : P_wrapper(NNinv_Torch=1, bottleneck_dim=10),
            # 'DBM_15' : P_wrapper(NNinv_Torch=1, bottleneck_dim=15),
            # 'DBM_20' : P_wrapper(NNinv_Torch=1, bottleneck_dim=20),

            # 'DBM_3_PCA' : P_wrapper(NNinv_Torch=1, bottleneck_dim=3, P='PCA'),
            # 'DBM_10_PCA' : P_wrapper(NNinv_Torch=1, bottleneck_dim=10, P='PCA'),
            # 'DBM_30_PCA' : P_wrapper(NNinv_Torch=1, bottleneck_dim=30, P='PCA'),

            # 'DBM_3_tSNE' : P_wrapper(NNinv_Torch=1, bottleneck_dim=3, P='tSNE'),
            # 'DBM_10_tSNE' : P_wrapper(NNinv_Torch=1, bottleneck_dim=10, P='tSNE'),
            # 'DBM_30_tSNE' : P_wrapper(NNinv_Torch=1, bottleneck_dim=30, P='tSNE'),
            }



###########################
#SAVE DIR
save_dir = './LID_results_new_grid500_GM'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


df = pd.DataFrame(columns=['Dataset', 'DM method', 'Dim', 'n_sample', 'n_cluster' ,'Intrinsic Dim (reconstructed)', 'Intrinsic Dim (data)'])
for data_name, dataset in datasets_real.items():
    print(data_name)
    print('='*20)
    print(f"processing {data_name}")
    X_train, X_test, y_train, y_test = dataset
    n_classes = len(np.unique(y_train))
    n_sample = X_train.shape[0]
    input_dim = X_train.shape[1]

    clf = LogisticRegression(max_iter=1000, random_state=420)
    clf.fit(X_train, y_train)
    print(f"training accuracy: {clf.score(X_train, y_train)}")
    # print(f"test accuracy: {clf.score(X_test, y_test)}")
    # data_eigen, data_id = get_eigen_general(X_train)
    # save_path = os.path.join(save_dir, f"{data_name}_data_eigen.npz")
    # np.savez(save_path, data_eigen=data_eigen)

    for proj_name, proj in projectors.items():
        print(f"processing {data_name} with {proj_name}")
        # print('='*20)   
        # ########## LID
        # lpca = skdim.id.lPCA(ver="Fan").fit_pw(X_train,
        #                       n_neighbors = 100,
        #                       n_jobs = -1)
        # print(lpca.dimension_pw_.shape)
        # print(np.mean(lpca.dimension_pw_))
        # # check bottleneck dim and the data dim
        if 'SSNP' not in proj_name:
            bottleneck_dim = proj.bottleneck_dim
            if bottleneck_dim > input_dim:
                print(f"bottleneck dim {bottleneck_dim} is larger than input dim {input_dim}, skip")
                continue
        
        if proj_name == 'DeepView':
            proj.fit(X_train, y_train, clf)
        else:
            proj.fit(X_train, y_train)

                 
        try:
            X_train_2d = proj.P.embedding_
        except:
            X_train_2d = proj.transform(X_train)
            print(proj_name, X_train_2d.shape)
        ##########################################
        X_recon = proj.inverse_transform(X_train_2d)
        #         ########## LID
        # lpca = skdim.id.lPCA(ver="Fan").fit_pw(X_recon,
        #                       n_neighbors = 200,
        #                       n_jobs = -1)
        # print(lpca.dimension_pw_.shape)
        # print(np.mean(lpca.dimension_pw_))
        # print('='*20)

        # xx, yy = np.meshgrid(np.linspace(X_train_2d[:, 0].min(), X_train_2d[:, 0].max(), 200),
        #                     np.linspace(X_train_2d[:, 1].min(), X_train_2d[:, 1].max(), 200))
        # grid = np.c_[xx.ravel(), yy.ravel()]
        # inv_grid = proj.inverse_transform(grid)
        # lid_map1 = skdim.id.lPCA().fit_pw( inv_grid, n_neighbors = 200,n_jobs=-1)
        # lid_map2 = skdim.id.lPCA(ver="Fan").fit_pw( inv_grid, n_neighbors = 200,n_jobs=-1)

        # map1 = lid_map1.dimension_pw_
        # map2 = lid_map2.dimension_pw_

        # map1 = map1.reshape(xx.shape)
        # map2 = map2.reshape(xx.shape)   

        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].contourf(xx, yy, map1, cmap='jet')
        # ax[1].contourf(xx, yy, map2, cmap='jet')
        # cbar1 = fig.colorbar(ax[0].contourf(xx, yy, map1, cmap='jet'), ax=ax[0])
        # cbar2 = fig.colorbar(ax[1].contourf(xx, yy, map2, cmap='jet'), ax=ax[1])
        # cbar1.ax.set_ylabel('LID_dfualt')
        # cbar2.ax.set_ylabel('LID_FAN')
        # plt.savefig(f'./figures/{data_name}_{proj_name}_LID.png')

        ########################################OLD 
        # map_builder = MapBuilder(clf, proj, X_train, y_train, grid=200)
        # alpha, labels = map_builder.get_prob_map()
        # # GM = map_builder.get_gradient_map()
        # GM = np.zeros(10)
        ##########################################NEW for gradient map
        GM, Iinv = get_gradient_map(projecters=proj, x2d=X_train_2d, grid=400)
        # print(GM.shape)
        probs = clf.predict_proba(Iinv)
        alpha = np.amax(probs, axis=1)
        labels = probs.argmax(axis=1)

        ########################
        if 'blob' in data_name:
            sampling_size = input_dim + 10
        else:
            sampling_size = 30
        # LID_finder = ID_finder_T(X_train_2d, proj, grid=500, sample_size=sampling_size, device='cuda', mode='nD', n_neighbors=120)
        # LID_evalues = LID_finder.LID_eval.to('cpu').numpy()
        LID_evalues = ID_finder_np(X_train_2d, proj, grid=500, sample_size=sampling_size, mode='nD').LID_eval

        ### save LID_evalues, (alpha, labels), GM, X_train_2d, y_train separately
        save_path = os.path.join(save_dir, f"{data_name}_{proj_name}.npz")
        np.savez(save_path, LID_evalues=LID_evalues, alpha=alpha, labels=labels, GM=GM, X_train_2d=X_train_2d, y_train=y_train, X_train=X_train, X_recon=X_recon)
        print(f"saved to {save_path}")
        

        # ##### another ID 
        # # rec_ID, data_ID = get_data_LID(X_train_2d, y_train, proj, device='cpu', data=X_train)
        # reco_eigen, reco_id = get_eigen_general(X_recon)
        # save_path = os.path.join(save_dir, f"{data_name}_{proj_name}_recon_eigen.npz")
        # np.savez(save_path, reco_eigen=reco_eigen)
        # ###
        # # get ID
        # if 'blob' in data_name:
        #     save_data_name = 'Blobs' + str(input_dim) + 'dims'

        # rec_id_mean = np.mean(reco_id)
        # data_id_mean = np.mean(data_id)
        # print(f"reconstructed ID: {rec_id_mean}")
        # print(f"data ID: {rec_id_mean}")
        # df = df._append({'Dataset': save_data_name, 'DM method': proj_name, 'Dim': input_dim ,'n_sample': n_sample, 'n_cluster': n_classes, 'Esimated ID (reco)': rec_id_mean, 'Esimated ID (data)': data_id_mean,
        #                  'bottleneck_dim': int(proj_name.split('_')[-1])}, ignore_index=True)
        # df.to_csv(os.path.join(save_dir, 'data_ID_bigneck.csv'), index=False)

# df.to_csv(os.path.join(save_dir, 'data_ID_results_moredata_har.csv'), index=False)




            

