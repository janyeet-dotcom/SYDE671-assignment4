import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from epipolar_utils import *

# Print 6 significant figures everywhere. Fixed-point with 6 decimals only shows
# 6 sig figs when |x| >= 0.1, so anything smaller (but non-zero) is printed in
# scientific notation instead, element by element.
SCI_THRESHOLD = 0.1


def _fmt_float(x):
    if x != 0 and abs(x) < SCI_THRESHOLD:
        return f"{x:.5e}"      # 1 digit + 5 decimals = 6 significant figures
    return f"{x:.6f}"


np.set_printoptions(linewidth=140, formatter={'float_kind': _fmt_float})


def report_svd(name, M):
    """Print the SVD breakdown of a 3x3 matrix and return (U, S, Vt)."""
    U, S, Vt = np.linalg.svd(M)
    print(f"  SVD of {name}:")
    print("    U =\n", _indent(U))
    print("    singular values S =", S)
    print("    Vt =\n", _indent(Vt))
    print(f"    det({name}) = {np.linalg.det(M):.6e}")
    print(f"    smallest/largest singular value = {S[2] / S[0]:.6e}")
    return U, S, Vt


def _indent(M, pad="      "):
    return "\n".join(pad + line for line in str(M).split("\n"))


'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    verbose - if True, print F and its SVD before/after the rank-2 enforcement
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
'''
def lls_eight_point_alg(points1, points2, verbose=False):
    n = points1.shape[0]
    # Each correspondence gives one row of W such that W f = 0, f = vec(F) (row-major)
    # p2^T F p1 = 0  ->  [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1] . f = 0
    W = np.zeros((n, 9))
    for i in range(n):
        u1, v1, _ = points1[i]
        u2, v2, _ = points2[i]
        W[i] = [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]

    # Least-squares solution of W f = 0, ||f|| = 1: right singular vector of the
    # smallest singular value
    _, _, Vt = np.linalg.svd(W)
    F_full = Vt[-1].reshape(3, 3)

    if verbose:
        print("  --- Rank-2 enforcement ---")
        print("  F BEFORE rank-2 enforcement:\n", _indent(F_full))
        U, S, Vt = report_svd("F (before)", F_full)
    else:
        U, S, Vt = np.linalg.svd(F_full)

    # Enforce rank 2: zero out the smallest singular value of F
    S_new = S.copy()
    S_new[2] = 0
    F = U @ np.diag(S_new) @ Vt

    if verbose:
        print(f"  Zeroing smallest singular value: {S[2]:.6e} -> 0")
        print("  F AFTER rank-2 enforcement:\n", _indent(F))
        report_svd("F (after)", F)
        print("  numerical rank before:", np.linalg.matrix_rank(F_full, tol=1e-12),
              "| after:", np.linalg.matrix_rank(F, tol=1e-10))
    return F


'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
'''
def normalized_eight_point_alg(points1, points2, verbose=False):
    def normalization_matrix(pts):
        # Translate centroid to origin, scale so mean distance from origin is sqrt(2)
        mean = pts[:, :2].mean(axis=0)
        d = np.sqrt(((pts[:, :2] - mean) ** 2).sum(axis=1)).mean()
        s = np.sqrt(2) / d
        return np.array([[s, 0, -s * mean[0]],
                         [0, s, -s * mean[1]],
                         [0, 0, 1]])

    T1 = normalization_matrix(points1)
    T2 = normalization_matrix(points2)
    q1 = (T1 @ points1.T).T
    q2 = (T2 @ points2.T).T

    if verbose:
        print("  T1 =\n", _indent(T1))
        print("  T2 =\n", _indent(T2))
        print("  Normalized points: mean =", q1[:, :2].mean(axis=0),
              "| mean distance from origin =",
              np.sqrt((q1[:, :2] ** 2).sum(axis=1)).mean(), "(target sqrt(2) = 1.414214)")

    F_hat = lls_eight_point_alg(q1, q2, verbose=verbose)  # includes rank-2 enforcement

    # Undo normalization: q2^T F_hat q1 = p2^T (T2^T F_hat T1) p1
    F = T2.T @ F_hat @ T1
    if verbose:
        print("  F in normalized coordinates (F_hat):\n", _indent(F_hat))
        print("  F after denormalization (T2^T F_hat T1):\n", _indent(F))
        print("  rank of denormalized F:", np.linalg.matrix_rank(F, tol=1e-10),
              "| det =", f"{np.linalg.det(F):.6e}")
    return F


'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F, title=None):

    def plot_epipolar_lines_on_image(points1, points2, im, F):
        im_height = im.shape[0]
        im_width = im.shape[1]
        lines = F.T.dot(points2.T)
        plt.imshow(im, cmap='gray')
        for line in lines.T:
            a,b,c = line
            xs = [1, im.shape[1]-1]
            ys = [(-c-a*x)/b for x in xs]
            plt.plot(xs, ys, 'r')
        for i in range(points1.shape[0]):
            x,y,_ = points1[i]
            plt.plot(x, y, '*b')
        plt.axis([0, im_width, im_height, 0])

    # We change the figsize because matplotlib has weird behavior when 
    # plotting images of different sizes next to each other. This
    # fix should be changed to something more robust.
    new_figsize = (8 * (float(max(im1.shape[1], im2.shape[1])) / min(im1.shape[1], im2.shape[1]))**2 , 6)
    fig = plt.figure(figsize=new_figsize)
    plt.subplot(121)
    plot_epipolar_lines_on_image(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_on_image(points2, points1, im2, F.T)
    plt.axis('off')
    if title:
        fig.suptitle(title)
    return fig


'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines (points1 vs. lines from points2).
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # Epipolar line in image 1 for each point in image 2: l = F^T p2 = (a, b, c)
    lines = F.T @ points2.T                      # 3 x N
    num = np.abs(np.sum(lines * points1.T, axis=0))   # |a*x + b*y + c|
    den = np.sqrt(lines[0] ** 2 + lines[1] ** 2)      # sqrt(a^2 + b^2)
    return np.mean(num / den)


def report_epipolar_lines(name, points1, points2, F, n_show=5):
    """Print epipolar line coefficients, per-point distances and summary stats."""
    lines = F.T @ points2.T                                   # lines in image 1
    dists = np.abs(np.sum(lines * points1.T, axis=0)) / np.sqrt(lines[0]**2 + lines[1]**2)
    lines_n = lines / np.sqrt(lines[0]**2 + lines[1]**2)      # (a,b) unit-normalised
    print(f"  [{name}] first {n_show} epipolar lines in image 1 (a, b, c), "
          f"a^2+b^2=1, with point-to-line distance:")
    for i in range(min(n_show, points1.shape[0])):
        a, b, c = lines_n[:, i]
        print(f"    point {i}: line = ({a:9.4f}, {b:9.4f}, {c:11.4f})   dist = {dists[i]:.4f} px")
    d1 = compute_distance_to_epipolar_lines(points1, points2, F)
    d2 = compute_distance_to_epipolar_lines(points2, points1, F.T)
    print(f"  [{name}] avg distance, image 1: {d1:.4f} px | image 2: {d2:.4f} px | "
          f"max single-point distance (image 1): {dists.max():.4f} px")
    return d1, d2


if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print('=' * 80)
        print("Set:", im_set)
        print('=' * 80)

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # ---------------- Linear least squares (un-normalized) ----------------
        print('\n' + '-' * 80)
        print("1) LLS eight-point algorithm (no normalization)")
        print('-' * 80)
        F_lls = lls_eight_point_alg(points1, points2, verbose=True)
        pFp_lls = np.abs([points2[i].dot(F_lls.dot(points1[i])) for i in range(points1.shape[0])])
        print("  max |p'^T F p| =", pFp_lls.max(), "| mean =", pFp_lls.mean())
        d_lls = report_epipolar_lines("LLS", points1, points2, F_lls)

        # ---------------- Normalized eight-point ----------------
        print('\n' + '-' * 80)
        print("2) Normalized eight-point algorithm")
        print('-' * 80)
        F_norm = normalized_eight_point_alg(points1, points2, verbose=True)
        pFp_n = np.abs([points2[i].dot(F_norm.dot(points1[i])) for i in range(points1.shape[0])])
        print("  max |p'^T F p| =", pFp_n.max(), "| mean =", pFp_n.mean())
        d_norm = report_epipolar_lines("Normalized", points1, points2, F_norm)

        # ---------------- Before vs after normalization ----------------
        print('\n' + '-' * 80)
        print("3) Comparison: before vs. after normalization")
        print('-' * 80)
        # Fix sign/scale ambiguity so matrices are comparable (unit Frobenius norm)
        A = F_lls / np.linalg.norm(F_lls)
        B = F_norm / np.linalg.norm(F_norm)
        if np.sum(A * B) < 0:
            B = -B
        print("  F (LLS), unit Frobenius norm:\n", _indent(A))
        print("  F (Normalized), unit Frobenius norm:\n", _indent(B))
        print(f"  {'':22s}{'LLS':>14s}{'Normalized':>14s}")
        print(f"  {'avg dist image 1 (px)':22s}{d_lls[0]:14.4f}{d_norm[0]:14.4f}")
        print(f"  {'avg dist image 2 (px)':22s}{d_lls[1]:14.4f}{d_norm[1]:14.4f}")
        print(f"  {'max |p2^T F p1|':22s}{pFp_lls.max():14.6f}{pFp_n.max():14.6f}")
        print(f"  {'mean |p2^T F p1|':22s}{pFp_lls.mean():14.6f}{pFp_n.mean():14.6f}")

        # ---------------- Epipolar line plots ----------------
        tag = im_set.split('/')[-1]
        fig1 = plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls,
                                             title=f"{tag}: LLS eight-point (avg dist {d_lls[0]:.2f}px / {d_lls[1]:.2f}px)")
        fig1.savefig(f"{tag}_epipolar_lls.png", dpi=150, bbox_inches='tight')
        fig2 = plot_epipolar_lines_on_images(points1, points2, im1, im2, F_norm,
                                             title=f"{tag}: Normalized eight-point (avg dist {d_norm[0]:.2f}px / {d_norm[1]:.2f}px)")
        fig2.savefig(f"{tag}_epipolar_normalized.png", dpi=150, bbox_inches='tight')
        print(f"\n  Saved figures: {tag}_epipolar_lls.png, {tag}_epipolar_normalized.png")

    plt.show()