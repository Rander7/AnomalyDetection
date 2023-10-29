import numpy as np
from sklearn.neighbors import LocalOutlierFactor
X = [[-1.1], [0.2], [101.1], [0.3]]
clf = LocalOutlierFactor(n_neighbors=2)
clf.fit_predict(X)

clf.negative_outlier_factor_



# --------------------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# 设置随机数生成器的种子，以便在每次运行程序时都能得到相同的随机数序列
# 可以确保不同的运行环境下得到相同的结果，方便调试和比较不同算法的性能
np.random.seed(42)

# 生成100个二维正态分布的样本，其中每个样本的坐标都乘以0.3，并分别加上2和减去2，以产生两个簇
# r_函数，它的作用是将两个数组沿着列方向（即第二个维度）拼接在一起，形成一个新的数组
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
# np.r_[X_inliers + 2, X_inliers - 2]将X_inliers数组中的每个元素都加上2和减去2，
# 然后将这两个结果数组沿着列方向拼接在一起，得到一个新的二维数组X_inliers，它的行数是原来的两倍，
# 每个元素都是在均值为2或-2、方差为0.09的正态分布中随机采样得到的
# 这个操作相当于在原始正常样本数据集中生成了两个距离较远的簇，用于模拟更加复杂的数据分布


# 生成20个二维均匀分布的样本，作为异常值
# 生成一个20行2列的二维数组，其中每个元素都是从均匀分布 (−4,4)中随机采样得到的
# 用于模拟数据集中的异常样本
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X_inliers, X_outliers]
# 将所有样本合并成一个数组，并为每个样本打上标签，其中正常样本标签为1，异常样本标签为-1
n_outliers = len(X_outliers)
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers:] = -1
# 第一行代码计算了异常样本的数量，即X_outliers数组的长度。
# 第二行代码生成了一个长度为len(X)的一维数组ground_truth，其中每个元素都是1。
# 这个数组表示了所有样本的真实标签，因为在这个例子中，所有样本都是正常样本，因此它们的真实标签都是1。
# 第三行代码将ground_truth数组中最后n_outliers个元素的值设置为-1，表示这些样本是异常样本。
# 具体来说，由于X_outliers是异常样本的特征矩阵，因此它的长度就是异常样本的数量。
# 因此，ground_truth[-n_outliers:] = -1将ground_truth数组中最后n_outliers个元素的值设置为-1，
# 表示这些样本是异常样本。这个操作相当于在原始正常样本数据集中添加了一些异常样本，用于模拟更加复杂的数据分布。

# 使用LOF算法拟合数据，并使用fit_predict方法计算每个样本的标签
# n_neighbors个最近邻居点的密度，并将该密度与它本身所在位置的密度进行比较，从而得到一个局部离群点因子
# 如果该因子小于1，则说明该样本周围的邻居点密度比较均匀，该样本是正常样本；
# 如果该因子大于1，则说明该样本周围的邻居点密度比较稀疏，该样本是异常样本
# contamination参数指定了数据集中异常样本的比例，它的默认值为0.1
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# 计算预测标签与真实标签不同的样本数量，并计算每个样本的离群点得分
# y_pred是一个一维数组，其中每个元素都是一个样本的局部离群点因子值
# 这些因子值可以用于判断每个样本是否为异常样本，但它们并没有直接被转化为1或-1的预测标签
# 预测标签是根据阈值和contamination参数计算得出的，具体来说，如果某个样本的局部离群点因子大于阈值，则被预测为异常样本，
# 标签为-1；否则被预测为正常样本，标签为1
# y_pred是LOF算法预测出的样本标签，而不是异常因子值
y_pred = clf.fit_predict(X)
n_errors = (y_pred != ground_truth).sum()
X_scores = clf.negative_outlier_factor_

# 画出所有样本点的散点图，并用圆圈表示每个样本的离群点得分，圆圈半径与得分成反比例关系

plt.title("Local Outlier Factor (LOF)")
plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
            facecolors='none', label='Outlier scores')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.xlabel("prediction errors: %d" % (n_errors))
legend = plt.legend(loc='upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()




# --------------------------------------------------------------------------------------------

# 新奇点检测
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

print(__doc__)

np.random.seed(42)

# 生成一个二维网格，其中每个网格点对应于平面上的一个坐标点。
# 具体来说，np.linspace(-5, 5, 500)生成了一个长度为500的等差数列，表示从-5到5之间的500个均匀间隔的数。
# 然后，np.meshgrid()函数将这两个等差数列作为参数，生成了一个二维网格。其中，xx和yy分别是两个二维数组，
# 它们的形状都是(500, 500)，表示在平面上生成了500*500=250000个坐标点。
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate normal (not abnormal) training observations
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate new normal (not abnormal) observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model for novelty detection (novelty=True)
clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
clf.fit(X_train)
# DO NOT use predict, decision_function and score_samples on X_train as this
# would give wrong results but only on new unseen data (not used in X_train),
# e.g. X_test, X_outliers or the meshgrid
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the learned frontier, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection with LOF")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                 edgecolors='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                edgecolors='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "errors novel regular: %d/40 ; errors novel abnormal: %d/40"
    % (n_error_test, n_error_outliers))
plt.show()
