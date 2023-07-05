from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    mean = x.mean(axis=0)
    x_cent = x - mean
    return x_cent

def get_covariance(dataset):
    dataset_transpose = np.transpose(dataset)
    cov = np.dot(dataset_transpose, dataset)/(dataset.shape[0]-1)
    return cov

def get_eig(S, m):
    eigen_values, eigen_vectors = eigh(S)
    return np.diag(np.flip(eigen_values[-m:])), np.flip(eigen_vectors[:,-m:], axis = 1)

def get_eig_prop(S, prop):
    eigen_values, eigen_vectors = eigh(S)
    idx = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    eigen_sum = sum(eigen_values)
    n = 0
    for eigenvalue in eigen_values:
        explained_variance = eigenvalue / eigen_sum
        if(explained_variance <= prop):
            break
        n = n + 1
        
    return np.diag(eigen_values[:n,]),eigen_vectors[:, :n]
    

def project_image(image, U):
    alpha =  image @ U
    return  np.expand_dims(alpha, axis = 0) @ U.T

def display_image(orig, proj):
    # Your implementation goes here!
    
    orig = np.reshape(orig,(32,32)).T
    proj = np.reshape(proj,(32,32)).T
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.set_title("Original")
    im1 = ax1.imshow(orig, aspect="equal")
    cbar1 = plt.colorbar(im1,ax=ax1)
    ax2.set_title("Projection")
    im2 = ax2.imshow(proj, aspect="equal")
    cbar2 = plt.colorbar(im2, ax=ax2)
    plt.show()
