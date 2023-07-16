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

# from ssnp import SSNP
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm

# import warnings
# warnings.filterwarnings('ignore')

# from map_evaluation import P_wrapper, MapBuilder

# sys.path.append('../GAN_inverse_projection')
# from utils import GANinv, CGANinv
# from umap import UMAP



class ID_finder_T:
    def __init__(self, X_2d, DM, grid=100, sample_size=5, device=None):
        self.LID_map = None
        self.DM = DM
        self.grid = grid
        self.sample_size = sample_size
        self.X_2d = X_2d
        if device is None:
            self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.LID_eval = self.get_LID()
        

    def get_LID(self):

        pixel_w = (self.X_2d[:, 0].max() - self.X_2d[:, 0].min()) / (self.grid-1)
        pixel_h = (self.X_2d[:, 1].max() - self.X_2d[:, 1].min()) / (self.grid-1)

        xx, yy = np.meshgrid(np.linspace(self.X_2d[:, 0].min(), self.X_2d[:, 0].max(), self.grid),
                            np.linspace(self.X_2d[:, 1].min(), self.X_2d[:, 1].max(), self.grid))
        XY = np.c_[xx.ravel(), yy.ravel()]

        data_shape = self.DM.inverse_transform(np.zeros((5 ,2)))
        LID_eval = T.zeros((XY.shape[0], data_shape.shape[1])).to(self.device)

        for i, pix in tqdm(enumerate(XY)):
            subset = self.get_subset(pix, pixel_w, pixel_h, self.sample_size)
            local_cov = self.compute_eigen(subset)
            LID_eval[i] = local_cov
        
        return LID_eval
    
    def get_subset(self, center, w, h, n_samples):
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

    def compute_eigen(self, subset):
        # to cuda, then compute cov
        subset = self.DM.inverse_transform(subset)
        subset = T.tensor(subset, dtype=T.float32, device=self.device)
        cov = T.cov(subset.T)

        ###
        regularization_factor = 1e-12
        cov += T.eye(cov.shape[0], device=self.device) * regularization_factor
        # comput eigenvalues
        eigvals = LA.eigvalsh(cov)
        sum_eigvals = T.sum(eigvals)
        eigvals = eigvals / sum_eigvals
        # reverse the order
        eigvals = T.flip(eigvals, dims=[0])        
        return eigvals

    def process_results(self, LID_eval, mode='dim', threshold=0.95):
        if mode == 'dim':
            # how many dimensions are needed to explain 95% of the variance    
            cumsum = T.cumsum(LID_eval, dim=1)
            LID_eval = (cumsum < threshold).sum(dim=1) + 1

        elif mode == 'percent':
            LID_eval = T.sum(LID_eval[:, :2], dim=1)
        else:
            raise ValueError('mode should be either dim or percent')
        return LID_eval
       
            

    def plot_LID(self,  ax=None,  cmap='jet', mode='dim', threshold=0.95):
        LID_map = self.process_results(self.LID_eval, mode, threshold)
        LID_map = LID_map.reshape(self.grid, self.grid).to('cpu').numpy()

        map = np.flip(LID_map, axis=0)

        if ax is None:
            # cbar of this plot
            if mode == 'dim':
                self.discrete_matshow(map, cmap=cmap)
            else: 
                image = plt.imshow(map, cmap=cmap, norm=MidpointNormalize(midpoint=0.95, vmax=1)) 
                # continuous colorbar
                plt.colorbar(image)
            # not show the ticks
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
                # continuous colorbar
                image = ax.imshow(map, cmap=cmap, norm=MidpointNormalize(midpoint=0.95, vmax=1))
                plt.colorbar(image, ax=ax)
                average = np.mean(map)
                ax.text(0.91, 0.05, f'{average:.2f}', color='white', transform=ax.transAxes)

            
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
