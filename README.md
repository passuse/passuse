from os import path
import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import *
from ipywidgets import interact

# 图像分割

# 首先，我们加载指纹图像:它将作为NumPy数组存储在内存中。
# %%
fingerprint = cv.imread('samples/sample_1_1.png', cv.IMREAD_GRAYSCALE)
show(fingerprint, f'Fingerprint with size (w,h): {fingerprint.shape[::-1]}')
# 计算局部梯度(使用Sobel滤波器)
gx, gy = cv.Sobel(fingerprint, cv.CV_32F, 1, 0), cv.Sobel(fingerprint, cv.CV_32F, 0, 1)
show((gx, 'Gx'), (gy, 'Gy'))
# 计算每个像素的梯度大小
gx2, gy2 = gx ** 2, gy ** 2
gm = np.sqrt(gx2 + gy2)
show((gx2, 'Gx**2'), (gy2, 'Gy**2'), (gm, 'Gradient magnitude'))
# 对方形窗口积分
sum_gm = cv.boxFilter(gm, -1, (25, 25), normalize=False)
show(sum_gm, 'Integral of the gradient magnitude')
# 使用一个简单的阈值分割指纹模式
thr = sum_gm.max() * 0.2
mask = cv.threshold(sum_gm, thr, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
show(fingerprint, mask, cv.merge((mask, fingerprint, fingerprint)))
# 步骤2:估计局部山脊方向
W = (23, 23)
gxx = cv.boxFilter(gx2, -1, W, normalize=False)
gyy = cv.boxFilter(gy2, -1, W, normalize=False)
gxy = cv.boxFilter(gx * gy, -1, W, normalize=False)
gxx_gyy = gxx - gyy
gxy2 = 2 * gxy

orientations = (cv.phase(gxx_gyy, -gxy2) + np.pi) / 2  # '-'调整y轴方向
sum_gxx_gyy = gxx + gyy
strengths = np.divide(cv.sqrt((gxx_gyy ** 2 + gxy2 ** 2)), sum_gxx_gyy, out=np.zeros_like(gxx), where=sum_gxx_gyy != 0)
show(draw_orientations(fingerprint, orientations, strengths, mask, 1, 16), 'Orientation image')
region = fingerprint[10:90, 80:130]
show(region)
smoothed = cv.blur(region, (5, 5), -1)
xs = np.sum(smoothed, 1)  # the x-signature of the region
print(xs)
x = np.arange(region.shape[0])
f, axarr = plt.subplots(1, 2, sharey=True)
axarr[0].imshow(region, cmap='gray')
axarr[1].plot(xs, x)
axarr[1].set_ylim(region.shape[0] - 1, 0)
plt.show()
# 求x签名局部最大值的索引
local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]
x = np.arange(region.shape[0])
plt.plot(x, xs)
plt.xticks(local_maxima)
plt.grid(True, axis='x')
plt.show()
# 计算连续峰值之间的所有距离
distances = local_maxima[1:] - local_maxima[:-1]
print(distances)
# 估计山脊线周期为上述距离的平均值
ridge_period = np.average(distances)
print(ridge_period)
# 创建过滤器组
or_count = 8
gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi / or_count)]
# 过滤整个图像与每个过滤器
nf = 255 - fingerprint
all_filtered = np.array([cv.filter2D(nf, cv.CV_32F, f) for f in gabor_bank])
show(nf, *all_filtered)
y_coords, x_coords = np.indices(fingerprint.shape)
# 对于每个像素，找到gabor bank中最近方向的索引
orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
# 取每个像素对应的卷积结果，组装最终结果
filtered = all_filtered[orientation_idx, y_coords, x_coords]
# 转换为灰度，并应用蒙版
enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)
show(fingerprint, enhanced)
# 微小位置的检测
# 二值化
_, ridge_lines = cv.threshold(enhanced, 32, 255, cv.THRESH_BINARY)
show(fingerprint, ridge_lines, cv.merge((ridge_lines, fingerprint, fingerprint)))
# 变薄
skeleton = cv.ximgproc.thinning(ridge_lines, thinningTy00pe=cv.ximgproc.THINNING_GUOHALL)
show(skeleton, cv.merge((fingerprint, fingerprint, skeleton)))


def compute_crossing_number(values):
    return np.count_nonzero(values < np.roll(values, -1))


# 创建一个过滤器，将任何8邻域转换为相应的字节值[0,255]
cn_filter = np.array([[1, 2, 4],
                      [128, 0, 8],
                      [64, 32, 16]
                      ])
# 创建一个查找表，将每个字节值映射到相应的交叉号
all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)
# 骨架:从0/255到0/1的值
skeleton01 = np.where(skeleton != 0, 1, 0).astype(np.uint8)
# 应用过滤器将每个像素的8个邻域编码为一个字节[0,255]
cn_values = cv.filter2D(skeleton01, -1, cn_filter, borderType=cv.BORDER_CONSTANT)
# 应用查找表获得每个像素的交叉数
cn = cv.LUT(cn_values, cn_lut)
# 只保留骨架上的交叉数字
cn[skeleton == 0] = 0
# 交叉数== 1 ->终止，交叉数== 3 ->分岔
minutiae = [(x, y, cn[y, x] == 1) for y, x in zip(*np.where(np.isin(cn, [1, 3])))]
show(draw_minutiae(fingerprint, minutiae), skeleton, draw_minutiae(skeleton, minutiae))
# 在计算距离变换之前，给蒙版添加一个1像素的背景边框
mask_distance = cv.distanceTransform(cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT), cv.DIST_C, 3)[1:-1, 1:-1]
show(mask, mask_distance)
filtered_minutiae = list(filter(lambda m: mask_distance[m[1], m[0]] > 10, minutiae))
show(draw_minutiae(fingerprint, filtered_minutiae), skeleton, draw_minutiae(skeleton, filtered_minutiae))


# 估计细节方向
def compute_next_ridge_following_directions(previous_direction, values):
    next_positions = np.argwhere(values != 0).ravel().tolist()
    if len(next_positions) > 0 and previous_direction != 8:
        #  有一个前一个方向:返回所有下一个方向，根据与它的距离排序，
        #  除了与前一个位置对应的方向(如果有的话)
        next_positions.sort(key=lambda d: 4 - abs(abs(d - previous_direction) - 4))
        if next_positions[-1] == (previous_direction + 4) % 8:  # 前一个位置的方向是相反的
            next_positions = next_positions[:-1]  # 删除它
    return next_positions


r2 = 2 ** 0.5  # sqrt(2)

# 八种可能的(x, y)偏移与每个相应的欧几里得距离
xy_steps = [(-1, -1, r2), (0, -1, 1), (1, -1, r2), (1, 0, 1), (1, 1, r2), (0, 1, 1), (-1, 1, r2), (-1, 0, 1)]

# LUT:对于每个8邻域和每个前一个方向[0,8]，
#      其中8表示“无”，提供了可能的方向列表
nd_lut = [[compute_next_ridge_following_directions(pd, x) for pd in range(9)] for x in all_8_neighborhoods]


def follow_ridge_and_compute_angle(x, y, d=8):
    px, py = x, y
    length = 0.0
    while length < 20:  # 最大长度
        next_directions = nd_lut[cn_values[py, px]][d]
        if len(next_directions) == 0:
            break
        # 需要检查所有可能的下一步方向
        if (any(cn[py + xy_steps[nd][1], px + xy_steps[nd][0]] != 2 for nd in next_directions)):
            break  # 又发现了一个细节:我们停在这里
        # 只需要遵循第一个方向
        d = next_directions[0]
        ox, oy, l = xy_steps[d]
        px += ox;
        py += oy;
        length += l
    # 检查是否已达到有效方向的最小长度
    return math.atan2(-py + y, px - x) if length >= 10 else None


valid_minutiae = []
for x, y, term in filtered_minutiae:
    d = None
    if term:  # 终止:简单跟随并计算方向
        d = follow_ridge_and_compute_angle(x, y)
    else:  # 分岔:遵循三个分支中的每一个
        dirs = nd_lut[cn_values[y, x]][8]  # 8表示:没有前一个方向
        if len(dirs) == 3:  # 除非正好有三个分支
            angles = [follow_ridge_and_compute_angle(x + xy_steps[d][0], y + xy_steps[d][1], d) for d in dirs]
            if all(a is not None for a in angles):
                a1, a2 = min(((angles[i], angles[(i + 1) % 3]) for i in range(3)),
                             key=lambda t: angle_abs_difference(t[0], t[1]))
                d = angle_mean(a1, a2)
    if d is not None:
        valid_minutiae.append((x, y, term, d))
show(draw_minutiae(fingerprint, valid_minutiae))
# 创建局部结构
# 计算一般局部结构的单元坐标
mcc_radius = 70
mcc_size = 16

g = 2 * mcc_radius / mcc_size
x = np.arange(mcc_size) * g - (mcc_size / 2) * g + g / 2
y = x[..., np.newaxis]
iy, ix = np.nonzero(x ** 2 + y ** 2 <= mcc_radius ** 2)
ref_cell_coords = np.column_stack((x[ix], x[iy]))
mcc_sigma_s = 7.0
mcc_tau_psi = 400.0
mcc_mu_psi = 1e-2


def Gs(t_sqr):
    """Gaussian function with zero mean and mcc_sigma_s standard deviation, see eq. (7) in MCC paper"""
    return np.exp(-0.5 * t_sqr / (mcc_sigma_s ** 2)) / (math.tau ** 0.5 * mcc_sigma_s)


def Psi(v):
    """Sigmoid function that limits the contribution of dense minutiae clusters, see eq. (4)-(5) in MCC paper"""
    return 1. / (1. + np.exp(-mcc_tau_psi * (v - mcc_mu_psi)))


# N:分钟数
# c:局部结构中的单元数

xyd = np.array([(x, y, d) for x, y, _, d in valid_minutiae])  # 包含所有细节坐标和方向的矩阵(n × 3)

# Rot: n × 2 × 2(每个细节的旋转矩阵)
d_cos, d_sin = np.cos(xyd[:, 2]).reshape((-1, 1, 1)), np.sin(xyd[:, 2]).reshape((-1, 1, 1))
rot = np.block([[d_cos, d_sin], [-d_sin, d_cos]])

# rot@ref_cell_coords。T: n × 2 × c
# xy : n x 2
xy = xyd[:, :2]
# cell_coordinates: n x c x 2(每个局部结构的单元格坐标)
cell_coords = np.transpose(rot @ ref_cell_coords.T + xy[:, :, np.newaxis], [0, 2, 1])

dists = np.sum((cell_coords[:, :, np.newaxis, :] - xy) ** 2, -1)

cs = Gs(dists)
diag_indices = np.arange(cs.shape[0])
cs[diag_indices, :, diag_indices] = 0

local_structures = Psi(np.sum(cs, -1))


@interact(i=(0, len(valid_minutiae) - 1))
def test(i=0):
    show(draw_minutiae_and_cylinder(fingerprint, ref_cell_coords, valid_minutiae, local_structures, i))


# 指纹比较
print(f"""Fingerprint image: {fingerprint.shape[1]}x{fingerprint.shape[0]} pixels
Minutiae: {len(valid_minutiae)}
Local structures: {local_structures.shape}""")
f1, m1, ls1 = fingerprint, valid_minutiae, local_structures
ofn = 'samples/sample_1_2'  # Fingerprint of the same finger
# ofn = 'samples/sample_2' # Fingerprint of a different finger
f2, (m2, ls2) = cv.imread(f'{ofn}.png', cv.IMREAD_GRAYSCALE), np.load(f'{ofn}.npz', allow_pickle=True).values()
dists = np.sqrt(np.sum((ls1[:, np.newaxis, :] - ls2) ** 2, -1))
dists /= (np.sqrt(np.sum(ls1 ** 2, 1))[:, np.newaxis] + np.sqrt(np.sum(ls2 ** 2, 1)))
# 选择距离最小的num_p对(LSS技术)
num_p = 5  # 简单起见:固定数量的配对
pairs = np.unravel_index(np.argpartition(dists, num_p, None)[:num_p], dists.shape)
score = 1 - np.mean(dists[pairs[0], pairs[1]])  # 见MCC论文式(23)
print(f'Comparison score: {score:.2f}')


@interact(i=(0, len(pairs[0]) - 1), show_local_structures=False)
def show_pairs(i=0, show_local_structures=False):
    show(draw_match_pairs(f1, m1, ls1, f2, m2, ls2, ref_cell_coords, pairs, i, show_local_structures))
