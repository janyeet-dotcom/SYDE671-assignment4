import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''

def factorization_method(points_im1, points_im2):
    # Measurement matrix D (2M x N): x and y coordinates of every point in each image.
    # Here M = 2 cameras, so D is 4 x N.
    D = np.vstack((points_im1[:, 0], points_im1[:, 1],
                   points_im2[:, 0], points_im2[:, 1]))

    # Center each row at the centroid of its points. This removes the translation of
    # the affine cameras, leaving D = (motion) * (structure) with rank 3.
    D = D - D.mean(axis=1, keepdims=True)

    # SVD of the centered measurement matrix
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    print("Singular values of D:", S)

    # Keep only the top 3 singular values (rank-3 factorization D ~ M S)
    motion = U[:, :3] @ np.diag(S[:3])    # 2M x 3 : camera (motion) matrix
    structure = Vt[:3, :]                 # 3 x N  : 3D structure
    return structure, motion

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        plt.show()
