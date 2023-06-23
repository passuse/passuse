# 导入所需的库
import cv2
import numpy as np
from skimage.filters._gabor import gabor_kernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# 添加matplotlib.pyplot和utils库
import matplotlib.pyplot as plt

# 定义指纹类别的标签
labels = {'Arch': 0, 'Tented Arch': 1, 'Left Loop': 2, 'Right Loop': 3, 'Whorl': 4}
# 定义数据集的路径
dataset_path = 'samples/'
# 定义训练集和测试集的文件名列表
train_files = ['DB1_B', 'DB2_B', 'DB3_B']
test_files = ['DB1_A', 'DB2_A', 'DB3_A']
# 定义一个过滤器，将任何8邻域转换为相应的字节值[0,255]
cn_filter = np.array([[1, 2, 4], [128, 0, 8], [64, 32, 16]])
# 创建一个查找表，将每个字节值映射到相应的交叉号
all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]




def compute_crossing_number(x):
    """
    Computes the crossing number of a given 8-neighbourhood.
    The crossing number is the number of times that adjacent pixels change value (from 0 to 1 or from 1 to 0).
    For example, the crossing number of [0, 0, 1, 1, 1, 0, 0, 0] is 2.
    """
    # 将第一个元素附加到列表的末尾以使其循环
    x = np.append(x, x[0])
    # 计算相邻元素之间的绝对差
    diff = np.abs(x[1:] - x[:-1])
    # 把这些差异加起来，然后除以二
    return np.sum(diff) // 2
cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)

# 定义一个函数，用于读取指纹图像和类别标签
# 定义一个函数，用于读取指纹图像和类别标签
def load_fingerprint_images_and_labels(files):
    images = []
    labels = []
    for file in files:
        # 读取文件中的每一行，每一行包含一个指纹图像的路径和类别标签
        with open(dataset_path + file + '.txt') as f:
            for line in f:
                # 分割路径和标签
                path, label = line.strip().split()
                # 读取灰度图像，将'.png'改为'.tif'
                image = cv2.imread(dataset_path + path + '.tif', cv2.IMREAD_GRAYSCALE)
                # 对图像进行旋转校正，使指纹的方向垂直于水平线
                angle = cv2.phase(image[0, 0], image[-1, 0])[0] * 180 / np.pi
                matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
                image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
                # 对图像进行裁剪，去除边缘部分
                image = cv2.getRectSubPix(image, (300, 300), (image.shape[1] / 2, image.shape[0] / 2))
                # 将图像和标签添加到列表中
                images.append(image)
                labels.append(label)
    # 将列表转换为数组并返回
    return np.array(images), np.array(labels)


# 调用函数，加载训练集和测试集
X_train, y_train = load_fingerprint_images_and_labels(train_files)
X_test, y_test = load_fingerprint_images_and_labels(test_files)


# 定义一个函数，用于提取指纹图像的特征向量
def extract_fingerprint_features(images):
    features = []
    for image in images:
        # 对图像进行二值化处理，使用Otsu’s方法自动选择阈值
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # 对图像进行细化处理，使用Zhang-Suen方法保留骨架结构
        image = cv2.ximgproc.thinning(image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        # 计算每个像素的8邻域的字节值[0,255]
        cn_values = cv2.filter2D(image // 255, -1, cn_filter, borderType=cv2.BORDER_CONSTANT)
        # 计算每个像素的交叉数，并将其转换为[0,1]范围内的值
        cn = cn_lut[cn_values] / 8.0
        # 估计指纹山脊线的周期
        smoothed = cv2.blur(image, (5, 5), -1) # 对图像进行平滑处理
        xs = np.sum(smoothed, 1) # 求x签名
        local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0] # 求x签名局部最大值的索引
        distances = local_maxima[1:] - local_maxima[:-1] # 计算连续峰值之间的所有距离
        ridge_period = np.average(distances) # 取平均值作为ridge_period
        # 对图像进行过滤处理，使用gabor_kernel函数创建一个过滤器组
        nf = 255 - image # 反转图像颜色
        or_count = 8 # 过滤器组的数量
        gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi / or_count)]
        all_filtered = np.array([cv2.filter2D(nf, cv2.CV_32F, f) for f in gabor_bank]) # 对图像进行卷积
        # 将过滤后的图像展平为一维向量，并添加到特征列表中
        features.append(all_filtered.flatten())
    # 将列表转换为数组并返回
    return np.array(features)



# 调用函数，提取训练集和测试集的特征向量
X_train = extract_fingerprint_features(X_train)
X_test = extract_fingerprint_features(X_test)
# 创建一个支持向量机分类器，使用径向基函数作为核函数
svm = SVC(kernel='rbf')
# 用训练集的特征向量和类别标签训练分类器
svm.fit(X_train, y_train)

# 用测试集的特征向量进行预测，并得到预测的类别标签
y_pred = svm.predict(X_test)

# 计算并打印预测的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
