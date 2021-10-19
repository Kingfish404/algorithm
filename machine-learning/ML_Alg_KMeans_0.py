"""二分类k均值算法演示
"""

import matplotlib.pyplot as plt
import numpy as np

n = 2  # 聚类的类数
num = 40  # 点的数目
size = 50  # 点的分布范围

X = np.random.rand(num) * size
Y = np.random.rand(num) * size

u = list()
for i in range(n):
    u.append(np.random.rand(2) * size)

u1 = np.array([20, 60])
u2 = np.array([80, 80])  # 迭代次数与初始点的选择好坏相关。

done = True
times = 1

while done:
    dist_1 = np.sqrt(np.power(X - u1[0], 2) + np.power(Y - u1[1], 2))
    dist_2 = np.sqrt(np.power(X - u2[0], 2) + np.power(Y - u2[1], 2))

    M1 = np.where(dist_1 < dist_2, 1, 0)
    M2 = np.where(dist_1 > dist_2, 1, 0)

    for j in range(len(M1)):
        M1[j] = j * M1[j]
        M2[j] = j * M2[j]

    newu1 = np.array([np.mean(X[M1]), np.mean(Y[M1])])
    newu2 = np.array([np.mean(X[M2]), np.mean(Y[M2])])

    distence = np.sqrt(np.sum((newu1 - u1) ** 2)) + \
               np.sqrt(np.sum((newu2 - u2) ** 2))

    done = True if distence > 1e-2 else False

    u1 = newu1
    u2 = newu2

    plt.cla()
    plt.title(str('The ' + str(times) + ' times KMeans'))
    plt.scatter(u1[0], u1[1], c="r")
    plt.annotate(str("N2"), xy=(u1[0], u1[1]), xytext=(u1[0] + 0.1, u1[1] + 0.1))

    plt.scatter(u2[0], u2[1], c="b")
    plt.annotate(str("N1"), xy=(u2[0], u2[1]), xytext=(u2[0] + 0.1, u2[1] + 0.1))

    plt.scatter(X[M1], Y[M1], s=10, c="r")
    plt.scatter(X[M2], Y[M2], s=10, c="b")
    for i in range(len(X)):
        # 这里xy是需要标记的坐标，xytext是对应的标签坐标
        plt.annotate(str(i), xy=(X[i], Y[i]), xytext=(X[i] + 0.1, Y[i] + 0.1))
    plt.waitforbuttonpress()
    times = times + 1

plt.cla()
plt.title(str('The ' + str(times) + ' times KMeans'))
plt.scatter(u1[0], u1[1], c="r")
plt.annotate(str("N2"), xy=(u1[0], u1[1]), xytext=(u1[0] + 0.1, u1[1] + 0.1))

plt.scatter(u2[0], u2[1], c="b")
plt.annotate(str("N1"), xy=(u2[0], u2[1]), xytext=(u2[0] + 0.1, u2[1] + 0.1))

plt.scatter(X[M1], Y[M1], s=10, c="r")
plt.scatter(X[M2], Y[M2], s=10, c="b")
for i in range(len(X)):
    # 这里xy是需要标记的坐标，xytext是对应的标签坐标
    plt.annotate(str(i), xy=(X[i], Y[i]), xytext=(X[i] + 0.1, Y[i] + 0.1))

plt.pause(10)
