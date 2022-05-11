import copy
import glob
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
# import torch
# from torchvision.transforms import ToTensor, ToPILImage
from scipy.spatial.distance import pdist, cdist, squareform

lk_params = dict(winSize=(9, 9),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01))

DEVICE = "cuda:0"


def Kalman_filter(x,P0=0.02,q=0.02,r=0.55):
	
	# 状态预测
    x_filter = x.copy()
    p_last = P0
    
    for i in range(1,x.shape[0]):
    	# 协方差预测公式
        p_mid = p_last + q
        # 卡尔曼系数
        k = p_mid / (p_mid + r)
        # 状态估计
        x_filter[i] = x_filter[i-1] + k * (x[i] - x_filter[i-1])
        # 噪声协方差更新
        p_last = (1 - k) * p_mid
        
    return x_filter

def harri_corners(image, mask):
    dst = cv2.cornerHarris(image, 3, 7, 0.04)
    # dst[mask == 0] = 0
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    # dst[mask == 0] = 0
    s = time.time()
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(image, np.float32(centroids), (5, 5), (-1, -1), criteria)
    print(time.time()-s)
    corners = corners.reshape((-1, 1, 2))
    return corners


def compute_size(image, offsets):
    # offsets (col, row)
    offsets = np.array(np.around(offsets), dtype=int)
    shape = image.shape
    cur_loc = copy.deepcopy(offsets[0])
    new_offsets = [offsets[0]]
    min_loc = 0
    max_loc = 0
    right_loc = cur_loc[0]
    for offset in offsets[1:]:
        cur_loc += offset
        new_offsets.append(copy.deepcopy(cur_loc))
        if min_loc > cur_loc[1]:
            min_loc = cur_loc[1]
        if max_loc < cur_loc[1]:
            max_loc = cur_loc[1]
        if right_loc < cur_loc[0]:
            right_loc = cur_loc[0]
    height = shape[0] + abs(min_loc) + max_loc
    width = shape[1] + right_loc
    return height + 2, width + 4, np.array(new_offsets) - np.array([-2, min_loc])


def stitch_image(image1, image2, offsets, mask):
    """
    :param image1: previous
    :param image2: current
    :param offsets: (col, row)
    :return: current image become previous
    """
    end_flag = False
    # corners = cv2.goodFeaturesToTrack(image1, 25, 0.3, 7, mask=mask)
    corners = harri_corners(image1, mask)
    p1, st, err = cv2.calcOpticalFlowPyrLK(image1, image2, corners, None, **lk_params)
    # offset_ = corners[st == 1] - p1[st == 1]
    corners = corners[st == 1]
    p1 = p1[st == 1]
    if len(corners) < 2:
        end_flag = True
    elif len(corners) < 4:
        offset_ = corners - p1
        offset_ = offset_[offset_[:, 0] > -1]
        offset = np.nanmedian(offset_, axis=0)
        if np.isnan(offset[0]) | np.isnan(offset[1]):
            end_flag = True
        else:
            offsets.append(offset)
    else:
        M, mask = cv2.findHomography(corners.reshape((-1, 1, 2)), p1.reshape((-1, 1, 2)), cv2.RANSAC, 5.0)
        mask = mask.ravel()
        if np.sum(mask) >= 2:
            offset_ = corners[mask == 1] - p1[mask == 1]
        else:
            offset_ = corners - p1
        offset_ = offset_[offset_[:, 0] > -1]
        offset = np.nanmedian(offset_, axis=0)
        if np.isnan(offset[0]) | np.isnan(offset[1]):
            end_flag = True
        else:
            offsets.append(offset)
    p1 = None
    return image2, offsets, p1, end_flag


def stitch(src_path):
    s = time.time()
    files = glob.glob(os.path.join(src_path, "*"))
    files = sorted(files)[18:]
    dst = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    mask = np.zeros(dst.shape[:2], dtype=np.uint8)
    mask[20:-20, :-100] = 255
    loc = []
    for idx, i in enumerate(files[1:]):
        cur_img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        dst, loc, cor, end_flag = stitch_image(dst, cur_img, loc, mask)
        if end_flag:
            break
    #kalman filter######################################
    loc_x = [xy[0] for xy in loc]
    data = np.array(loc_x)
    kalman_loc_x = Kalman_filter(data ,P0=0.02,q=0.02,r=0.55) # to do kalman filter
    #plt.plot(loc_x, label="normal")
  #  plt.plot(kalman_loc_x, label="kalman")
  #  plt.legend()
  #  plt.show()
    loc_y = [xy[1] for xy in loc]
    loc = [[x, y] for x, y in zip(kalman_loc_x, loc_y)]
    ####################################################
    res = compute_size(dst, offsets=loc)
    result = np.zeros((res[0], res[1], 3), dtype=np.uint8)
    for f, loc in zip(files[1:], res[2]):
        # print(f, loc)
        img = cv2.imread(f)
        result[loc[1]:loc[1] + img.shape[0], loc[0] + 10:loc[0] + img.shape[1], :] = img[:, 10:]
    result = simple_spline(res[2], result, dst.shape[0])
    # src_points, dst_points = get_match_points(res[2], result)
    # result = tps_transform(src_points, dst_points, result)
    # start_row = src_points[0][1]
    # end_row = start_row + dst.shape[0] - 10
    # result = result[start_row:end_row, :, :]
    print(f"time elapse {time.time()-s}s")
    plt.imshow(result)
    plt.show()
    return result


def simple_spline(src_points, result, height):
    x = np.arange(result.shape[1])
    y = np.interp(x, src_points[:, 0], src_points[:, 1])
    rr = np.zeros_like(result)
    for i in range(result.shape[1]):
        rr[:height, i, :] = result[int(y[i]):int(y[i]) + height, i, :]
    rr = rr[:height, :, :]
    return rr


# class TPS(torch.nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, X, Y, w, h, device):
#         """
#         :param X: target points [1, 8, 2] 8代表控制点的个数，2代表维度，这里只有两维
#         :param Y: src points [1, 8, 2]
#         :param w: width
#         :param h: height
#         :param device: cpu or gpu
#         :return:
#         """
#         """ 计算grid"""
#         grid = torch.ones(1, h, w, 2, device=device)
#         grid[:, :, :, 0] = torch.linspace(-1, 1, w)
#         grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
#         grid = grid.view(-1, h * w, 2)
#
#         """ 计算W, A"""
#         n, k = X.shape[:2]
#         device = X.device
#
#         Z = torch.zeros(1, k + 3, 2, device=device)
#         P = torch.ones(n, k, 3, device=device)
#         L = torch.zeros(n, k + 3, k + 3, device=device)
#
#         eps = 1e-9
#         D2 = torch.pow(X[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
#         K = D2 * torch.log(D2 + eps)
#
#         P[:, :, 1:] = X
#         Z[:, :k, :] = Y
#         L[:, :k, :k] = K
#         L[:, :k, k:] = P
#         L[:, k:, :k] = P.permute(0, 2, 1)
#
#         Q = torch.solve(Z, L)[0]
#         W, A = Q[:, :k], Q[:, k:]
#
#         """ 计算U """
#         eps = 1e-9
#         D2 = torch.pow(grid[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
#         U = D2 * torch.log(D2 + eps)
#
#         """ 计算P """
#         n, k = grid.shape[:2]
#         device = grid.device
#         P = torch.ones(n, k, 3, device=device)
#         P[:, :, 1:] = grid
#
#         # grid = P @ A + U @ W
#         grid = torch.matmul(P, A) + torch.matmul(U, W)
#         return grid.view(-1, h, w, 2)
#
#
# def norm_torch(points_int, width, height):
#     """
#     将像素点坐标归一化至 -1 ~ 1
#     """
#     points_int_clone = torch.from_numpy(points_int).detach().float().to(DEVICE)
#     x = ((points_int_clone * 2)[..., 0] / (width - 1) - 1)
#     y = ((points_int_clone * 2)[..., 1] / (height - 1) - 1)
#     return torch.stack([x, y], dim=-1).contiguous().view(-1, 2)
#
#
# def tps_transform_torch(source, target, image):
#     ten_img = ToTensor()(image).to(DEVICE)
#     h, w = ten_img.shape[1], ten_img.shape[2]
#     ten_source = norm_torch(source, w, h)
#     ten_target = norm_torch(target, w, h)
#     tps = TPS()
#     warped_grid = tps(ten_target[None, ...], ten_source[None, ...], w, h, DEVICE)  # 这个输入的位置需要归一化，所以用norm
#     ten_wrp = torch.grid_sampler_2d(ten_img[None, ...], warped_grid, 0, 0, True)
#     new_img_torch = np.array(ToPILImage()(ten_wrp[0].cpu()))
#     return new_img_torch


def get_match_points(locations, image):
    shape = image.shape
    locations = np.unique(locations, axis=0)
    locations = locations[::10, :]
    target_points = copy.deepcopy(locations)
    target_points[:, 1] = target_points[0][1]
    bottom_source = copy.deepcopy(locations)
    bottom_source[:, 1] = bottom_source[:, 1] + shape[0]
    source = np.r_[locations, bottom_source]
    bottom_target = copy.deepcopy(target_points)
    bottom_target[:, 1] = bottom_target[:, 1] + shape[0]
    target = np.r_[target_points, bottom_target]
    return source, target


def makeT(cp):
    # cp: [K x 2] control points
    # T: [(K+3) x (K+3)]
    K = cp.shape[0]  # 获取控制点的个数
    T = np.zeros((K + 3, K + 3))
    T[:K, 0] = 1
    T[:K, 1:3] = cp
    T[K, 3:] = 1
    T[K + 1:, 3:] = cp.T
    R = squareform(pdist(cp, metric='euclidean'))
    R = R * R
    R[R == 0] = 1  # a trick to make R ln(R) 0
    R = R * np.log(R)
    np.fill_diagonal(R, 0)
    T[:K, 3:] = R
    return T


def liftPts(p, cp):
    # p: [N x 2], input points
    # cp: [K x 2], control points
    # pLift: [N x (3+K)], lifted input points
    N, K = p.shape[0], cp.shape[0]
    pLift = np.zeros((N, K + 3))
    pLift[:, 0] = 1
    pLift[:, 1:3] = p
    R = cdist(p, cp, 'euclidean')
    R = R * R
    R[R == 0] = 1
    R = R * np.log(R)
    pLift[:, 3:] = R
    return pLift


def norm(points, width, height):
    col = points[:, 0] * 2 / (width - 1) - 1
    row = points[:, 1] * 2 / (height - 1) - 1
    points = np.c_[col, row]
    return points


def tps_transform(target, source, image):
    # source control points
    cps = norm(source, image.shape[1], image.shape[0])
    # target control points
    target = norm(target, image.shape[1], image.shape[0])
    xt = target[:, 0]  # 获取col
    yt = target[:, 1]  # 获取row
    # construct T
    T = makeT(cps)
    # solve cx, cy (coefficients for x and y)
    xtAug = np.concatenate([xt, np.zeros(3)])
    ytAug = np.concatenate([yt, np.zeros(3)])
    cx = np.linalg.solve(T, xtAug)  # [K+3]
    cy = np.linalg.solve(T, ytAug)

    # dense grid
    x = np.linspace(-1, 1, image.shape[1])
    y = np.linspace(-1, 1, image.shape[0])
    x, y = np.meshgrid(x, y)
    xgs, ygs = x.flatten(), y.flatten()
    gps = np.vstack([xgs, ygs]).T

    # transform
    pgLift = liftPts(gps, cps)  # [N x (K+3)]
    xgt = np.dot(pgLift, cx.T)
    ygt = np.dot(pgLift, cy.T)
    points_x = (xgt + 1) * (image.shape[1] - 1) / 2
    points_y = (ygt + 1) * (image.shape[0] - 1) / 2
    x = np.array(np.around(points_x), dtype=np.uint)
    y = np.array(np.around(points_y), dtype=np.uint)

    x = x.reshape((image.shape[0], image.shape[1]))
    y = y.reshape((image.shape[0], image.shape[1]))
    x[x < 0] = 0
    x[x >= image.shape[1]] = image.shape[1] - 1
    y[y < 0] = 0
    y[y >= image.shape[0]] = image.shape[0] - 1
    new_image = image[y, x, :]
    return new_image


if __name__ == '__main__':
    np.set_printoptions(suppress=True,
                        precision=10,
                        threshold=2000,
                        linewidth=150)
    # super_fold = r"D:\workspace\stitch\images"
    # folds = glob.glob(os.path.join(super_fold, "*"))
    # for fold in folds:
    #     if os.path.isdir(fold):
    #         sub_folds = glob.glob(os.path.join(fold, "*"))
    #         for sub_fold in sub_folds:
    #             if os.path.isdir(sub_fold):
    #                 print(sub_fold)
    #                 dst_path = os.path.join(os.path.dirname(sub_fold), "new_"+os.path.basename(sub_fold)+".png")
    #                 try:
    #                     image = stitch(sub_fold)
    #                     cv2.imwrite(dst_path, image)
    #                 except Exception as e:
    #                     print(e, sub_fold)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # folds = glob.glob(os.path.join(r"D:\workspace\stitch\scan_pen_20220117-3", "*_bmp"))
    # for f in folds:
    #     print(f)
    #     if os.path.isdir(f):
    #         dst_path = os.path.join(os.path.dirname(f), "new_" + os.path.basename(f) + ".png")
    #         # dst_path = f + ".png"
    #         files = glob.glob(os.path.join(f, "*"))
    #         if len(files) > 10:
    #             try:
    #                 image = stitch(f)
    #                 cv2.imwrite(dst_path, image)
    #             except:
    #                 print("error", f)

    image = stitch(r"/home/ps/DiskA/project/dxf-pdf/cc_bmp/usbRgbPicSave-90_bmp")
