import numpy as np
from scipy.linalg import null_space
import pandas as pd
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist
#from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from tqdm import tqdm

import os  
os.chdir(os.path.dirname(__file__))


def euclidean_mds(data):
    mds = MDS()
    transformed = mds.fit_transform(data)
    stress = mds.stress_
    return transformed, stress

#NN calculation
def get_k_nearest_neighbors_hd(point_hd, data, k=1):
    point_hd = point_hd.reshape(1, -1) 
    dist = np.sum((data - point_hd)**2, axis=1)
    indices_nn = np.argsort(dist)[:k]
    return indices_nn


#point selection strategies for multilateration
def get_furthest_points(data, reduced_data, p):
    '''
    function to get the n+1 furthest points and their distances (in reduced space) 
    to the point p 
    
    '''
    n = data.shape[1] #dimensionality in high dimensional space
    d = cdist(reduced_data, p) #distances to p
    combine = np.hstack((data,d))
    sorted_combine = combine[np.argsort(combine[:,-1])] #sort according to distances
    #return n+1 furthest points
    furthest_d = sorted_combine[-(n+1):,-1].reshape(-1,1) 
    furthest_P = sorted_combine[-(n+1):,:-1] 
    return furthest_P, furthest_d

def get_nearest_points(data, reduced_data, p):
    '''
    function to get the n+1 nearest points and their distances (in reduced space) 
    to the point p 
    
    '''
    n = data.shape[1] #dimensionality in high dimensional space
    d = cdist(reduced_data, p) #distances to p
    combine = np.hstack((data,d)) 
    sorted_combine = combine[np.argsort(combine[:,-1])] #sort according to distances
    #return n+1 nearest points (exluding p itself if p is existing point)
    if np.any([np.array_equal(p, point) for point in reduced_data]):
        nearest_d = sorted_combine[1:n+2, -1].reshape(-1, 1) # exclude p
        nearest_P = sorted_combine[1:n+2, :-1]
    else:
        nearest_d = sorted_combine[:n+1,-1].reshape(-1,1) 
        nearest_P = sorted_combine[:n+1,:-1] 
    return nearest_P, nearest_d

def get_random_points(data, reduced_data, p):
    '''
    function to get random n+1 points and their distances (in reduced space) 
    to the point p 
    
    '''
    n = data.shape[1] #dimensionality in high dimensional space
    #return n+1 random points
    random_indices = np.random.choice(data.shape[0], n+1, replace=False) # randomly select indices
    while np.any([np.array_equal(p, point) for point in reduced_data[random_indices]]): # ensure p is not a selected point 
        random_indices = np.random.choice(data.shape[0], n+1, replace=False)
    random_P = data[random_indices] 
    random_d = cdist(reduced_data[random_indices], p).reshape(-1, 1) # distances to p
    return random_P, random_d

def get_kmedoids_points(data, reduced_data, p):
    '''
    function to get k-medoids cluster centers and their distances (in reduced space) 
    to the point p
    '''
    n = data.shape[1]  
    # apply k-medoids clustering and get medoids
    #kmedoids = KMedoids(n_clusters=n+1, random_state=0).fit(reduced_data)
    #medoids_indices = kmedoids.medoid_indices_
    kmeans = KMeans(n_clusters=n+1).fit(data) #.fit(reduced_data)
    cluster_centers = kmeans.cluster_centers_

    medoids_indices = []
    for center in cluster_centers:
        distances = cdist(data, center.reshape(1,-1)) #reduced_data
        medoid_index = np.argmin(distances) #nearest point to centroid (medoid)
        if np.array_equal(reduced_data[medoid_index],p): # ensure p is not a selected point 
            medoid_index = np.argsort(distances)[1, :] #take second nearest point to centroid if medoid is p
        medoids_indices.append(medoid_index)
    
    medoids_P = data[medoids_indices]
    medoids_d = cdist(reduced_data[medoids_indices], p).reshape(-1, 1) # distances to p
    return medoids_P, medoids_d

def get_random_cluster_points(data, reduced_data, p, kmeans):
     '''
     function to get n+1 points and their distances (in reduced space) 
     to the point p where each point comes from another cluster
     '''
     cluster_centers = kmeans.cluster_centers_

     indices = []
     for center in cluster_centers:
        cluster_points = np.where(kmeans.labels_ == np.where(cluster_centers == center)[0][0])[0]
        random_index = np.random.choice(cluster_points)
        while np.array_equal(reduced_data[random_index],p): # ensure p is not a selected point 
            random_index = np.random.choice(cluster_points)
        indices.append(random_index)
         
     P = data[indices]
     d = cdist(reduced_data[indices], p).reshape(-1, 1) # distances to p
     return P, d   
 
#multilateration    
def multilateration(P,d):
    '''
    function to get the position of a point by using the distances d of the
    points P to this point

    '''
    A = (P[1:,:] - P[0,:]) * (-2)
    e = d**2 - np.sum(P**2, axis = 1).reshape(-1,1)
    B = e[1:] - e[0]
    if (np.linalg.det(A) != 0): #full rank -> only 1 exact solution
        p_particular = np.linalg.solve(A, B).T
    else: #solution with min l2 norm
        p_particular = np.linalg.lstsq(A.T, B, rcond=None)[0] #B.T
        #all solutions: p_particular + c * null_space
    return p_particular


def multilateration2(P, d):
    '''
    function to get the position of a point by using the distances d of the
    points P to this point

    '''
    A = (P[1:,:] - P[0,:]) * (-2)
    e = d**2 - np.sum(P**2, axis = 1).reshape(-1,1)
    B = e[1:] - e[0]
    try:
        return np.linalg.solve(A, B).T
    except:
        return np.linalg.lstsq(A.T, B, rcond=None)[0]


#inverse projection 
def get_any_high_dimensional_position(data, reduced_data, point, point_selection='random', trials = 20, scaling_factor = 1):
    '''
    function to estimate the position of a point in n-dimensional space
    using multilateration

    Parameters
    ----------
    data : ndarray
        data in high-dimensional space.
    reduced_data : ndarray
        data in low-dimensional space.
    point : arry
        arbitrary point in low-dimensional space.
    point_selection: str
        point selection strategy for multilateration. Default is random.
    trials: int
        number of trials for the random point selection strategy. Default is 20.
    scaling_factor: float
        approx. scaling factor of MDS. Default is 1 (perfect distance preservation).

    Returns
    -------
    position : np.array - (n x 1) vector
        estimated position of point in high-dimensional space.

    '''
    point = point.reshape(1,-1)
    #get n+1 points according to point selection strategy
    if point_selection == 'furthest':
        P, d = get_furthest_points(data, reduced_data, point)
        d = scaling_factor*d
        position = multilateration(P, d)
    elif point_selection == 'nearest':
        P, d = get_nearest_points(data, reduced_data, point)
        d = scaling_factor*d
        position = multilateration(P, d)
    elif point_selection == 'kmedoids':
        P, d = get_kmedoids_points(data, reduced_data, point)
        d = scaling_factor*d
        position = multilateration(P, d)
    elif point_selection == 'random_with_clusters':
        '''
        P, d = get_random_cluster_points(data, reduced_data, point)
        position = multilateration(P, d)
        '''
        n = data.shape[1]  
        # clustering in reduced space
        kmeans = KMeans(n_clusters=n+1).fit(reduced_data)
        positions = []
        random_points = []
        for i in range(trials):
            random_points.append(get_random_cluster_points(data, reduced_data, point, kmeans))
        for e in random_points:
            positions.append(multilateration2(e[0], e[1]*scaling_factor).flatten())
        positions = np.array(positions)
        position = positions[np.argmin(cdist(positions, positions).sum(axis=0))] # get medoid of generated positions  
    else: #random
        positions = []
        random_points = []
        for i in range(trials):
            random_points.append(get_random_points(data, reduced_data, point))
        for e in random_points:
            positions.append(multilateration2(e[0], e[1]*scaling_factor).flatten())
        positions = np.array(positions)
        #position = np.mean(positions, axis=0) # get centroid of randomly generated positions
        #position = np.median(positions, axis=0) # get median of randomly generated positions
        position = positions[np.argmin(cdist(positions, positions).sum(axis=0))] # get medoid of generated positions
    return position


class MDSinv:
    def __init__(self,  point_selection='random', trials = 20) -> None:
        self.point_selection = point_selection
        self.trials = trials

    def fit(self, X2d, X, **kwarg):
        self.X = X
        self.X2d = X2d

    def transform(self, p_list, **kwarg):
        # p_list = np.array(p_list)
        # v_func = np.vectorize(get_any_high_dimensional_position)
        # return v_func(self.X, self.X2d, p_list, self.point_selection, self.trials)
        Xnd_recons = []
        for p in tqdm(p_list):
            pnd = get_any_high_dimensional_position(self.X, self.X2d, p, self.point_selection, self.trials)
            Xnd_recons.append(pnd.reshape(-1))
        print(len(Xnd_recons))
        print(Xnd_recons[0].shape)
        print(Xnd_recons[2].shape)
        return np.array(Xnd_recons).reshape(-1, self.X.shape[1])
    
    def inverse_transform(self, p, **kwarg):
        return self.transform(p, **kwarg)
    
    def reset(self):
        self.X = None
        self.X2d = None
