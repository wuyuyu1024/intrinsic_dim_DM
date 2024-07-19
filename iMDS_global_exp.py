import sys
import os

sys.path.append('../inverse_projections')


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
#import knn and decision tree
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE, MDS
from multilateration import MDSinv

from LID import ID_finder_T, get_data_LID, get_eigen_general , ID_finder_np, ID_finder_T
from gradient_map import get_gradient_map


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
# # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]) # Notice here
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)


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
    'blobs_dim10_n5000_y10',
    
    'blobs_dim30_n5000_y10',
    # 'blobs_dim60_n5000_y10',
    'blobs_dim100_n5000_y10',

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
            # 'SDBM' : P_wrapper(ssnp=1),
            # 'DBM': P_wrapper(NNinv_Torch=1, ),
            # 'DeepView': P_wrapper(deepview=1),
            # 'UMAP+iLAMP': Simple_P_wrapper(UMAP(random_state=42), Pinv_ilamp()),
            # 'UMAP+RBF': Simple_P_wrapper(UMAP(random_state=42), RBFinv()),
            'MDS+iMDS': Simple_P_wrapper(MDS(n_components=2, random_state=42), MDSinv()),
}



save_dir = './LID_results_new_grid500_GM_iMDS_global'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



for data_name, dataset in datasets_real.items():
    print(data_name)
    print('='*20)
    print(f"processing {data_name}")
    X_train, X_test, y_train, y_test = dataset
    n_classes = len(np.unique(y_train))
    n_sample = X_train.shape[0]
    input_dim = X_train.shape[1]

    clf = linear_model.LogisticRegression(max_iter=1000, random_state=420)
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
      
        ##########################################NEW for gradient map
        GM, Iinv = get_gradient_map(projecters=proj, x2d=X_train_2d, grid=500)
        # print(GM.shape)
        probs = clf.predict_proba(Iinv)
        alpha = np.amax(probs, axis=1)
        labels = probs.argmax(axis=1)

        save_path = os.path.join(save_dir, f"{data_name}_{proj_name}.npz")
        np.savez(save_path, alpha=alpha, labels=labels, GM=GM, X_train_2d=X_train_2d, y_train=y_train, X_train=X_train, X_recon=X_recon)

        print(f"partially saved to {save_path}")
        print('ID estimation....')
        eigen_list = ID_finder_T.compute_eigen(data=Iinv)
        eigen_list_cpu = eigen_list.cpu().numpy()

        ### save LID_evalues, (alpha, labels), GM, X_train_2d, y_train separately
        save_path = os.path.join(save_dir, f"{data_name}_{proj_name}.npz")
        np.savez(save_path, eigen_list_global=eigen_list_cpu, alpha=alpha, labels=labels, GM=GM, X_train_2d=X_train_2d, y_train=y_train, X_train=X_train, X_recon=X_recon)
        print(f"saved to {save_path}")