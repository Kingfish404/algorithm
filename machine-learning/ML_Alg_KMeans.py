"""二分类k均值算法演示
"""

import matplotlib.pyplot as plt
import numpy as np

n = 3  # 聚类的类数
num = 9  # 点的数目
size = 50  # 点的分布范围

X = np.random.rand(num) * size
Y = np.random.rand(num) * size

u = list()  # 各个聚类的中心点
for i in range(n):
    u.append(np.random.rand(2) * size)

done = True
times = 1

while done:
    dist = list()
    N = list()
    for i in range(n):
        out = np.sqrt(np.power(X - u[i][0], 2) + np.power(Y - u[i][1], 2))
        dist.append(out.reshape(1, num))

    TheMin = np.array(dist).min(0)[0]
    M = np.array([0] * n * num)
    M = M.reshape(n, num)
    NPDist = np.squeeze(np.array(dist)).reshape(n, num)
    for i in range(num):
        b = TheMin[i]
        for j in range(n):
            a = NPDist[j][i]
            if a - b < 1e-5:
                M[j][i] = 1
                break

    new = list()
    for i in range(n):
        sumX = 0
        sumY = 0
        k = 0
        for j in range(num):
            sumX = sumX + X[i] * M[i][j]
            sumY = sumY + Y[i] * M[i][j]
            k = k + M[i][j]
        new.append([sumX / k, sumY / k])

    u = new.copy()
    print(u)

    plt.cla()
    plt.title(str('The ' + str(times) + ' times KMeans'))
    for i in range(n):
        plt.scatter(u[i][0], u[i][1],s=200, c="b")
        plt.annotate("N" + str(i), xy=(u[i][0], u[i][1]), xytext=(u[i][0] + 0.1, u[i][1] + 0.1))

    plt.scatter(X, Y, s=10, c="r")
    for i in range(num):
        # 这里xy是需要标记的坐标，xytext是对应的标签坐标
        for j in range(n):
            if(M[j][i] == 1):
                plt.scatter(X[i],Y[i], c="C"+str(j))
                plt.annotate(str(i), xy=(X[i], Y[i]), xytext=(X[i] + 0.1, Y[i] + 0.1))
    plt.waitforbuttonpress()

    times = times + 1