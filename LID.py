# import os
# import sys
# sys.path.append('../sdbm/code')
# sys.path.append('../DeepView/deepview')
# sys.path.append('../dbm_evaluation')
# import make blobs
# from sklearn.datasets import make_blobs
# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# import matplotlib as mpl
import numpy as np
# import pandas as pd
# import seaborn as sns
import torch as T
import torch.linalg as LA

# imort find neighbors
from sklearn.neighbors import NearestNeighbors


# from ssnp import SSNP
# from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm

# import warnings
# warnings.filterwarnings('ignore')

# from map_evaluation import P_wrapper, MapBuilder

# sys.path.append('../GAN_inverse_projection')
# from utils import GANinv, CGANinv
# from umap import UMAP

class ID_finder_np:
    def __init__(self, X_2d, DM, grid=100, sample_size=5, mode='2D'):
        self.LID_map = None
        self.DM = DM
        self.grid = grid
        self.sample_size = sample_size
        self.X_2d = X_2d
        self.LID_eval = self.get_LID(mode=mode)

    def get_LID(self, mode='2D'):
        assert mode in ['2D', 'nD']

        pixel_w = (self.X_2d[:, 0].max() - self.X_2d[:, 0].min()) / (self.grid-1)
        pixel_h = (self.X_2d[:, 1].max() - self.X_2d[:, 1].min()) / (self.grid-1)

        xx, yy = np.meshgrid(np.linspace(self.X_2d[:, 0].min(), self.X_2d[:, 0].max(), self.grid),
                             np.linspace(self.X_2d[:, 1].min(), self.X_2d[:, 1].max(), self.grid))
        XY = np.c_[xx.ravel(), yy.ravel()]

        data_shape = self.DM.inverse_transform(np.zeros((5 ,2)))
        n_dim = data_shape.shape[1]
        LID_eval = np.zeros((XY.shape[0], n_dim))

        if mode == 'nD':
            self.pixel_inv = self.DM.inverse_transform(XY)
            self.fnn = NearestNeighbors(n_neighbors=120, algorithm='ball_tree', n_jobs=-1).fit(self.pixel_inv)
        for i, pix in tqdm(enumerate(XY)):
            if mode == '2D':
                subset = self.get_subset_2d(pix, pixel_w, pixel_h, self.sample_size)
                local_cov = self.compute_eigen(subset, self.DM)
            elif mode == 'nD':
                nd_center = self.pixel_inv[i]
                subdata = self.get_subset_nd(nd_center)
                # print('~~~~~~~~~~~~~~~~~~~~~~~~~!!!!!!!!!!!!!!!1')
                # print('shape of subdata:', subdata.shape)

                local_cov = self.compute_eigen(data=subdata)

            LID_eval[i] = local_cov
        
        return LID_eval

    def get_subset_2d(self, center, w, h, n_samples):
        x, y = center
        lower = [x - w/2, y - h/2]
        upper = [x + w/2, y + h/2]
        samples = np.random.uniform(lower, upper, (n_samples, 2))
        return samples
    
    def get_subset_nd(self, nd_center, radius=None):
        subset_ind = self.fnn.kneighbors(nd_center.reshape(1, -1), return_distance=False)
        subset = self.pixel_inv[subset_ind].squeeze(0)
        # scaler = StandardScaler()
        # subset = scaler.fit_transform(subset)
        return subset

    @staticmethod
    def compute_eigen(subset=None, Pinv=None, data=None):
        if data is None:
            subset = Pinv.inverse_transform(subset)
            cov = np.cov(subset, rowvar=False)
        else:
            cov = np.cov(data, rowvar=False)

        regularization_factor = 1e-12
        cov += np.eye(cov.shape[0]) * regularization_factor
        eigvals = np.linalg.eigvalsh(cov)
        sum_eigvals = np.sum(eigvals)
        eigvals = eigvals / sum_eigvals
        eigvals = np.flip(eigvals)
        return eigvals

    @staticmethod
    def process_results(LID_eval, mode='dim', threshold=0.95):
        if mode == 'TV':
            cumsum = np.cumsum(LID_eval, axis=1)
            LID_eval = (cumsum < threshold).sum(axis=1) + 1
        elif mode == 'MV':
            LID_eval = (LID_eval >= (1-threshold)).sum(axis=1)
        elif mode == 'VR':
            delta_v = LID_eval[:, :] - np.concatenate((LID_eval[:, 1:], np.zeros((LID_eval.shape[0], 1))), axis=1)
            total_dv = np.sum(delta_v, axis=1)
            norm_delta_v = delta_v/total_dv.reshape(-1, 1)
            cumsum = np.cumsum(norm_delta_v, axis=1)
            LID_eval = (cumsum < threshold).sum(axis=1) + 1
        elif mode == 'percent':
            LID_eval = np.sum(LID_eval[:, :2], axis=1)
        else:
            raise ValueError('mode should be either dim or percent')
        return LID_eval

    def plot_LID(self, ax=None, cmap='jet', mode='dim', threshold=0.95):
        LID_map = self.process_results(self.LID_eval, mode, threshold)
        LID_map = LID_map.reshape(self.grid, self.grid)

        map = np.flip(LID_map, axis=0)
        if ax is None:
            if mode == 'dim':
                self.discrete_matshow(map, cmap=cmap)
            else: 
                image = plt.imshow(map, cmap=cmap)
                plt.colorbar(image)
            plt.xticks([])
            plt.yticks([])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            if mode == 'dim':
                ax = self.discrete_matshow(map, cmap=cmap, ax=ax)
                average = np.mean(map)
                ax.set_title(f'Average LID: {average:.2f}')
                ax.text(0.91, 0.05, f'{average:.2f}', color='w', transform=ax.transAxes)
            else:
                image = ax.imshow(map, cmap=cmap)
                plt.colorbar(image, ax=ax)
                average = np.mean(map)
                ax.text(0.91, 0.05, f'{average:.2f}', color='white', transform=ax.transAxes)




class ID_finder_T:
    def __init__(self, X_2d, DM, grid=100, sample_size=5, device=None, mode='2D', n_neighbors=120):
        self.LID_map = None
        self.DM = DM
        self.grid = grid
        self.sample_size = sample_size
        self.X_2d = X_2d
        if device is None:
            self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.n_neighbors = n_neighbors
        self.LID_eval = self.get_LID(mode=mode)
        

    def get_LID(self, mode='2D'):
        ## assert mode
        assert mode in ['2D', 'nD']

        pixel_w = (self.X_2d[:, 0].max() - self.X_2d[:, 0].min()) / (self.grid-1)
        pixel_h = (self.X_2d[:, 1].max() - self.X_2d[:, 1].min()) / (self.grid-1)

        xx, yy = np.meshgrid(np.linspace(self.X_2d[:, 0].min(), self.X_2d[:, 0].max(), self.grid),
                            np.linspace(self.X_2d[:, 1].min(), self.X_2d[:, 1].max(), self.grid))
        XY = np.c_[xx.ravel(), yy.ravel()]

        data_shape = self.DM.inverse_transform(np.zeros((5 ,2)))
        n_dim = data_shape.shape[1]
        LID_eval = T.zeros((XY.shape[0], n_dim)).to(self.device)

        if mode == 'nD':
            self.pixel_inv = self.DM.inverse_transform(XY)
            self.fnn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(self.pixel_inv)
        for i, pix in tqdm(enumerate(XY)):
            if mode == '2D':
                subset = self.get_subset_2d(pix, pixel_w, pixel_h, self.sample_size)
                local_cov = self.compute_eigen(subset, self.DM, self.device)
            elif mode == 'nD':
                nd_center = self.pixel_inv[i]
                subdata = self.get_subset_nd(nd_center)
                # print('~~~~~~~~~~~~~~~~~~~~~~~~~!!!!!!!!!!!!!!!1')
                # print('shape of subdata:', subdata.shape)
                # print('AAAAAAAAAA@@@@@@@@@@~~~~~!!!!!!!!!!!!!!!1')
                local_cov = self.compute_eigen(data=subdata)

            LID_eval[i] = local_cov
        
        return LID_eval
    
    def get_subset_2d(self, center, w, h, n_samples):
        """
        Generate `n_samples` uniformly distributed within a square of `size` around `center`.

        Parameters:
        center (tuple): The center of the square (x, y).
        size (float): The size of the square. Points will be generated in the range [center - size/2, center + size/2].
        n_samples (int): The number of samples to generate.

        Returns:
        numpy.ndarray: An array of shape (n_samples, 2) containing the generated points.
        """
        x, y = center
        lower = [x - w/2, y - h/2]
        upper = [x + w/2, y + h/2]

        samples = np.random.uniform(lower, upper, (n_samples, 2))
        return samples
    
    def get_subset_nd(self, nd_center, radius=None):
        # get a subset of data within a fixed neighborhood
        subset_ind = self.fnn.kneighbors(nd_center.reshape(1, -1), return_distance=False)
        subset = self.pixel_inv[subset_ind].squeeze(0)
        # normalize the subset
        # print('shape of subset:', subset.shape)
        # print(subset.min(), subset.max())
        # scaler = StandardScaler()
        # subset = scaler.fit_transform(subset)

        return subset
        

    @staticmethod
    def compute_eigen(subset=None, Pinv=None, device=None, data=None, normalize=True):
        if device is None:
            device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        # to cuda, then compute cov
        if data is None:
            subset = Pinv.inverse_transform(subset)
            subset = T.tensor(subset, dtype=T.float32, device=device)
            cov = T.cov(subset.T)
        else:
            data = T.tensor(data, dtype=T.float32, device=device)
            cov = T.cov(data.T)

        ### add regularization
        regularization_factor = 1e-8
        cov += T.eye(cov.shape[0], device=device) * regularization_factor
        ############################
        # comput eigenvalues
        eigvals = LA.eigvalsh(cov)
        if normalize:
            sum_eigvals = T.sum(eigvals)
            eigvals = eigvals / sum_eigvals
        # reverse the order
        eigvals = T.flip(eigvals, dims=[0])        
        return eigvals

    @staticmethod
    def process_results(LID_eval, mode='TV', threshold=0.95):
        if type(LID_eval) == np.ndarray:
            LID_eval = T.tensor(LID_eval)
        if mode == 'TV':
            # how many dimensions are needed to explain 95% of the variance    
            cumsum = T.cumsum(LID_eval, dim=1)
            LID_eval = (cumsum < threshold).sum(dim=1) + 1
        elif mode == 'MV':
            ## number of dimnesions larger than 5% of the variance
            # print(LID_eval)
            LID_eval = (LID_eval >= (1-threshold)).sum(dim=1) 
            # print(LID_eval)
        elif mode == 'VR':
            ## compute the variance ratio
            delta_v = LID_eval[:, ] - T.concatenate((LID_eval[:, 1:], T.zeros((LID_eval.shape[0], 1))), dim=1)
            total_dv = T.sum(delta_v, dim=1)
            norm_delta_v = delta_v/total_dv.reshape(-1, 1)
            cumsum = T.cumsum(norm_delta_v, dim=1)
            LID_eval = (cumsum < threshold).sum(dim=1) + 1   
        elif mode == 'percent':
            LID_eval = T.sum(LID_eval[:, :2], dim=1)
        else:
            raise ValueError('mode should be either the above 4: TV, MV, VR, percent ')
        return LID_eval   
            

    def plot_LID(self,  ax=None,  cmap='jet', mode='dim', threshold=0.95):
        LID_map = self.process_results(self.LID_eval, mode, threshold)
        LID_map = LID_map.reshape(self.grid, self.grid).to('cpu').numpy()

        map = np.flip(LID_map, axis=0)
        if ax is None:
            if mode == 'percent': 
                image = plt.imshow(map, cmap=cmap)
                plt.colorbar(image)
            else:
                self.discrete_matshow(map, cmap=cmap)
            plt.xticks([])
            plt.yticks([])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            
            if mode == 'percent':
                image = ax.imshow(map, cmap=cmap)
                plt.colorbar(image, ax=ax)
                average = np.mean(map)
                ax.text(0.91, 0.05, f'{average:.2f}', color='white', transform=ax.transAxes)
            else:
                ax = self.discrete_matshow(map, cmap=cmap, ax=ax)
                average = np.mean(map)
                ax.set_title(f'Average LID: {average:.2f}')
                ax.text(0.91, 0.05, f'{average:.2f}', color='w', transform=ax.transAxes)

            
    def discrete_matshow(self, data, cmap, ax=None):
    # get discrete colormap
        cmap = plt.get_cmap(cmap, np.max(data) - np.min(data) + 1)
        # set limits .5 outside true range
        if ax is None:
            mat = plt.imshow(data, cmap=cmap, vmin=np.min(data) - 0.5, 
                            vmax=np.max(data) + 0.5)
            # tell the colorbar to tick at integers
            cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data) + 1))
        else:
            mat = ax.imshow(data, cmap=cmap, vmin=np.min(data) - 0.5, 
                            vmax=np.max(data) + 0.5)
            
            # tell the colorbar to tick at integers
            fig = ax.get_figure()
            # cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
            plt.colorbar(mat,  ticks=np.arange(np.min(data), np.max(data) + 1))
            return ax
            
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y)) 


def get_data_LID(X_2d, y, Pinv, threshold=0.95, device='cpu', data=None):
    n_class = len(np.unique(y))
    reco_lid_list = []
    data_lid_list = []
    for i in range(n_class):
        subset = X_2d[y==i]
        eigen_val = ID_finder_T.compute_eigen(subset, Pinv, device=device)
        ## reshape eigen_val
        eigen_val = eigen_val.reshape(1, -1)
        lid = ID_finder_T.process_results(eigen_val, mode='dim', threshold=threshold).to('cpu').numpy()
        reco_lid_list.append(lid)

        if data is not None:
            data_subset = data[y==i]
            eigen_val = ID_finder_T.compute_eigen(None, None, device=device, data=data_subset)
            ## reshape eigen_val
            eigen_val = eigen_val.reshape(1, -1)
            lid = ID_finder_T.process_results(eigen_val, mode='dim', threshold=threshold).to('cpu').numpy()
            data_lid_list.append(lid)
    # return the average LID
    if data is None:
        return np.mean(reco_lid_list)
    else:
        return np.mean(reco_lid_list), np.mean(data_lid_list)
    
def get_eigen_general(X, n_neighbors=100, GPU=False, mode='TV', threshold=0.95, normalize=True):
    fnn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1).fit(X)
    eigen_list = []
    id_list = []
    for i in tqdm(X):
        subset_ind = fnn.kneighbors(i.reshape(1, -1), return_distance=False)
        subset = X[subset_ind].squeeze(0)
        dev = T.device('cuda' if T.cuda.is_available() else 'cpu')
        if GPU:
            cov = ID_finder_T.compute_eigen(data=subset, device=dev, normalize=normalize)
            eigen_list.append(cov.to('cpu').numpy())
        else:
            cov = ID_finder_np.compute_eigen(data=subset)
            eigen_list.append(cov)

    eigen_list = np.array(eigen_list)
    if not GPU:
        lid_list = ID_finder_np.process_results(eigen_list, mode=mode, threshold=threshold)
    elif normalize and GPU:
        lid_list = ID_finder_T.process_results(T.tensor(eigen_list).to(dev), mode=mode, threshold=threshold).to('cpu').numpy()
    else:
        ## No. eigen that > 1 
        lid_list = (eigen_list > 1).sum(axis=1)
    return eigen_list, lid_list
        

