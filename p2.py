import numpy as np
import matplotlib.pyplot as plt
from p1 import *
from epipolar_utils import *

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    # The epipole e in image 2 satisfies e^T F = 0, i.e. F^T e = 0, so e spans the
    # null space of F^T. With F^T = U S V^T, that is the right singular vector of
    # the smallest singular value (last row of V^T).
    _, _, Vt = np.linalg.svd(F.T)
    e = Vt[-1]
    return e / e[2]  # homogeneous coordinates [x y 1]

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    height, width = im.shape[0], im.shape[1]
    e = e / e[2]

    # 1. Translation: move the image centre to the origin
    T = np.array([[1, 0, -width / 2.0],
                  [0, 1, -height / 2.0],
                  [0, 0, 1]])
    e_t = T @ e                      # translated epipole (e1, e2, 1)
    e1, e2 = e_t[0], e_t[1]

    # 2. Rotation: put the translated epipole on the x-axis at (f, 0, 1)
    alpha = 1.0 if e1 >= 0 else -1.0
    norm = np.sqrt(e1 ** 2 + e2 ** 2)
    R = np.array([[ alpha * e1 / norm, alpha * e2 / norm, 0],
                  [-alpha * e2 / norm, alpha * e1 / norm, 0],
                  [0, 0, 1]])
    f = (R @ e_t)[0]

    # 3. Send (f, 0, 1) to (f, 0, 0): a point at infinity on the horizontal axis
    G = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [-1.0 / f, 0, 1]])

    # 4. Undo the translation
    return np.linalg.inv(T) @ G @ R @ T

'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    e2 = e2 / e2[2]
    H2 = compute_H(e2, im2)

    # Cross-product matrix [e]_x, so that [e]_x v = e x v
    e_x = np.array([[0, -e2[2], e2[1]],
                    [e2[2], 0, -e2[0]],
                    [-e2[1], e2[0], 0]])
    v = np.array([[1.0, 1.0, 1.0]])
    M = e_x @ F + np.outer(e2, v)

    # H0 = H2 M maps image 1 into the rectified frame (up to the free shear H_A)
    H0 = H2 @ M
    p1_hat = (H0 @ points1.T).T
    p1_hat /= p1_hat[:, 2:3]
    p2_hat = (H2 @ points2.T).T
    p2_hat /= p2_hat[:, 2:3]

    # Least squares W a = b: choose (a1, a2, a3) so that x-coordinates line up
    W = p1_hat                       # rows [x1_hat, y1_hat, 1]
    b = p2_hat[:, 0]                 # x2_hat
    a, _, _, _ = np.linalg.lstsq(W, b, rcond=None)

    H_A = np.array([[a[0], a[1], a[2]],
                    [0, 1, 0],
                    [0, 0, 1]])
    H1 = H_A @ H0
    return H1, H2


if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    # F is such that such that (points2)^T * F * points1 = 0, so e1 is e' and e2 is e
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print('')

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
