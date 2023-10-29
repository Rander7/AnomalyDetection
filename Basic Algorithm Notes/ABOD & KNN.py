# 模拟数据集中的PyOD
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.data import generate_data, get_outliers_inliers

# 使用generate_data函数生成具有两个特征的随机数据。
# n_train参数指定生成的训练数据数量，train_only参数设置为True表示仅生成训练数据，n_features参数指定数据的特征数量
X_train, Y_train = generate_data(n_train=200, train_only=True, n_features=2)

# 默认情况下，generate_data函数将异常比例设置为0.1。
outlier_fraction = 0.1

# 将生成的数据集分为异常点和正常点，并分别存储在x_outliers和x_inliers两个numpy数组中
x_outliers, x_inliers = get_outliers_inliers(X_train, Y_train)

# 计算正常点和异常点的数量
n_inliers = len(x_inliers)
n_outliers = len(x_outliers)

# 将数据集的两个特征分离出来，分别存储在F1和F2两个numpy数组中，以便后续绘图使用
# 将X_train数据集中的第一个特征（列索引为0）提取出来，并将其形状重新整形为1列的数组。
# X_train[:,[0]]：使用冒号（:）表示选取所有行，[0]表示选取列索引为0的特征。因此，这个表达式表示从X_train数据集中提取出第一个特征的所有值。
# .reshape(-1,1)：使用reshape函数将提取出的特征数组进行形状重塑。
# -1表示自动计算行数，1表示指定为1列。因此，这个表达式将特征数组从原来的形状重塑为只有一列的数组。
F1 = X_train[:, [0]].reshape(-1, 1)
F2 = X_train[:, [1]].reshape(-1, 1)

# 使用np.meshgrid函数创建一个网格，用于绘制散点图
xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))

# 使用scatter函数绘制散点图，横轴和纵轴分别表示F1和F2特征，添加x轴和y轴的标签
plt.scatter(F1, F2)
plt.xlabel('F1')
plt.ylabel('F2')
# plt.show()

# 通过键来访问对应的异常检测算法的实例。这样做的好处是可以方便地在代码中切换和比较不同的异常检测算法
classifiers = {
     'Angle-based Outlier Detector (ABOD)'   : ABOD(contamination=outlier_fraction),
     'K Nearest Neighbors (KNN)' :  KNN(contamination=outlier_fraction)
     # 'LOF' : LOF(contamination=outlier_fraction)
}
# set the figure size
plt.figure(figsize=(10, 10))

# 使用enumerate函数来遍历classifiers字典中的键值对，并为每个键值对分配一个索引
for i, (clf_name, clf) in enumerate(classifiers.items()):
    # fit the dataset to the model
    clf.fit(X_train)

    # decision_function是该实例的一个方法，用于计算给定数据集的异常分数
    # 乘以 - 1的操作是为了将异常分数转换为负数。这是因为在一些异常检测算法中，异常分数越低表示样本越异常
    scores_pred = clf.decision_function(X_train) * -1
    # print(scores_pred)

    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X_train)

    # no of errors in prediction
    n_errors = (y_pred != Y_train).sum()
    print('No of Errors : ', clf_name, n_errors)

    # 可视化

    # 阈值决定数据点是被视为内点还是离群值
    # scoreatpercentile函数用于计算给定数据集在指定百分位数处的分数。
    # scores_pred是异常分数的数组，100 * outlier_fraction是一个百分位数，表示异常样本的比例乘以100。
    # 这个百分位数表示将异常样本的异常分数作为阈值的分数。
    # threshold：这是一个变量，用于存储计算得到的阈值。
    threshold = stats.scoreatpercentile(scores_pred, 100 * outlier_fraction)
    # print(threshold)

    # 决策函数计算每个点的原始异常分数
    # 给定数据集的异常分数是根据整个数据集计算出的，而每个点的原始异常分数是根据该点与其他点的相对关系计算出来的
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)
    # 使用clf对象的decision_function方法计算输入数据的异常分数，并将结果存储在Z变量中。np.c_函数用于将xx
    # 和yy两个数组按列连接成一个二维数组，其中xx.ravel()和yy.ravel()分别将xx和yy数组展平成一维数组。
    # 这样就可以将每个坐标点作为输入数据传递给decision_function方法进行计算。由于decision_function
    # 方法计算的是每个点相对于决策边界的距离，因此需要将结果乘以 - 1，以便将距离转换为异常分数。最终得到的
    # Z数组是一个二维数组，其中包含了每个坐标点的异常分数

    subplot = plt.subplot(1, 2, i + 1)

    # 将蓝色色图从最小异常分数填充到阈值值
    subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 10), cmap=plt.cm.Blues_r)
    # contourf()函数将二维数组`Z`中的异常分数可视化为填充的等高线图，并将其添加到`subplot`变量所指向的子图中。其中
    # `xx`和`yy`分别是沿x轴和y轴的坐标点数组，用于指定等高线图的位置。`levels`参数用于指定等高线图的分层级别，`np.linspace()`
    # 函数用于在最小异常分数和阈值之间创建10个等距级别的数组。这些级别将用于将异常分数映射到填充的颜色图中。
    # `cmap`参数指定使用的颜色图
    # `plt.cm.Blues_r`是一种蓝色调色板，用于将低异常分数映射到浅蓝色，高异常分数映射到深蓝色。最终结果是一个填充的等高线图，
    # 其中颜色越深表示异常分数越高，即越接近阈值。

    # 异常分数等于阈值时绘制红色等高线
    a = subplot.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')

    # 在异常分数范围从阈值到最大异常时绘制橙色等高线
    subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')

    # 绘制内点的散点图，使用白色的点
    b = subplot.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], c='white', s=20, edgecolor='k')
    # 绘制外点的散点图，使用黑色的点
    c = subplot.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], c='black', s=20, edgecolor='k')
    subplot.axis('tight')

    subplot.legend(
        [a.collections[0], b, c],
        ['learned decision function', 'true inliers', 'true outliers'],
        prop=matplotlib.font_manager.FontProperties(size=10),
        loc='lower right')

    subplot.set_title(clf_name)
    subplot.set_xlim((-10, 10))
    subplot.set_ylim((-10, 10))
plt.show()
