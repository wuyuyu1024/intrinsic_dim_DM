import os
import sys
sys.path.append('../sdbm/code')
sys.path.append('../DeepView/deepview')
sys.path.append('../GAN_inverse_projection')
# import make blobs
from sklearn.datasets import make_blobs
# import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import torch as T
import torch.linalg as LA
from tqdm import tqdm

from ssnp import SSNP
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

from map_evaluation import P_wrapper

class ID_finder:
    def __init__(self):
        self.LID_map = None
        self.DM = None
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    def get_LID(self, DM, grid=100, sample_size=5, mode='dim', cmap='jet'):
        self.DM = DM

        pixel_w = 1 / grid
        xx, yy = np.meshgrid(np.linspace(0, 1, grid), np.linspace(0, 1, grid))
        XY = np.c_[xx.ravel(), yy.ravel()]

        LID_map = np.zeros(XY.shape[0])
        data_shape = DM.inverse_transform(np.zeros((5 ,2)))
        # print(data_shape)
        LID_eval = T.zeros((XY.shape[0], data_shape.shape[1])).to(self.device)

        for pix in tqdm(XY):
            subset = self.get_subset(pix, pixel_w, sample_size)
            local_cov = self.compute_cov(subset)
            LID_eval[pix] = local_cov
            # value = self.process_results(local_cov, mode)
            # LID_map[pix] = value
        LID_map = self.process_results(LID_eval, mode)
        self.LID_map = LID_map.reshape(grid, grid).to('cpu').numpy()
        self.LID_eval = LID_eval.to('cpu').numpy()

    def get_subset(self, center, size, n_samples):
        # def sample_uniform_2d(center, size, n_samples):
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
        lower = [x - size/2, y - size/2]
        upper = [x + size/2, y + size/2]

        samples = np.random.uniform(lower, upper, (n_samples, 2))
        return samples

    def compute_cov(self, subset):
        # to cuda, then compute cov
        # subset = 
        # do not show the warning form tensorflow
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        subset = self.DM.inverse_transform(subset)
        subset = T.tensor(subset, dtype=T.float32, device=self.device)
        cov = T.cov(subset.T)
        # comput eigenvalues
        eigvals = LA.eigvalsh(cov)
        # order the eigenvalues
        # eigvals = T.sort(eigvals, descending=True)

        return eigvals

    def process_results(self, LID_eval, mode='dim'):
        ## sort the eigenvalues, 
        LID_eval, ind = T.sort(LID_eval, descending=True)
        # if mode == 'dim', return how many dimensions are needed to explain 95% of the variance; if mode == 'percent', return the percentage of variance explained by the first 2 dimensions
        ### LID_eval: (n_samples, n_features)
        if mode == 'dim':
            # compute the cumulative sum
            cumsum = T.cumsum(LID_eval, dim=1)
            # find the index of the first element that is larger than 0.95
            index = T.argmax((cumsum > 0.95).long(), dim=1)
            # return the number of dimensions
            return index + 1
        elif mode == 'percent':
            # compute the percentage of variance explained by the first 2 dimensions
            return LID_eval[:, 0] / T.sum(LID_eval, dim=1)
                
    
            

    def plot_LID(self, DM=None, ax=None, grid=50, cmap='jet', mode='dim', sample_size=5):
        if DM != self.DM or self.LID_map is None:
            self.DM = DM
            # set the others to None
            self.get_LID(grid=grid, DM=DM, mode=mode, sample_size=sample_size)

        map = np.flip(self.LID_map, axis=0)
        # map = self.LID_map
        if ax is None:
            plt.imshow(map, cmap=cmap)
            # cbar of this plot
            plt.colorbar()

        else:
            ax.imshow(map, cmap=cmap)
            # ax.colorbar()
            # cbar of this plot
            fig = ax.get_figure()
            cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
            norm = mpl.colors.Normalize(vmin=map.min(), vmax=map.max())
            mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
            



X, y = make_blobs(n_samples=200, centers=3, n_features=3, random_state=0)
X = MinMaxScaler().fit_transform(X)
clf = LogisticRegression(random_state=0).fit(X, y)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# ssnp = SSNP(bottleneck_activation='linear', verbose=0)
# ssnp.fit(X, y, )
projecter = P_wrapper(deepview=1, ssnp=0, NNinv_Torch=0)
projecter.fit(X, y, clf)

X_2d = projecter.transform(X)

lid_finder = ID_finder()
# lid_finder.get_LID(ssnp, grid=10, mode='dim')
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
print('plotting...')
lid_finder.plot_LID(DM=projecter, grid=10, mode='dim', ax=ax)
print('done')

plt.show()


