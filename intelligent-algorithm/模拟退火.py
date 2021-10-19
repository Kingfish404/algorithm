import numpy as np
import matplotlib.pyplot as plt
'''
模拟退火优化算法
'''
class SAoptimizer:
    def __init__(self):
        super().__init__()

    def optimize(self, f, ybound=(-np.inf, np.inf), initf=np.random.random, randf=np.random.random,
            t=10000, alpha=0.98, stop=1e-1, iterPerT=1, l=1):
        '''
        :param f: 目标函数,接受np.array作为参数 :param ybound: y取值范围 
        :param initf: 目标函数的初始权值函数，返回np.array :param alpha: 退火速率  
        :param iterPerT: 每个温度下迭代次数 :param t: 初始温度 :param l:新旧值相减后的乘数，越大，越不容易接受更差值
        :param stop: 停火温度 :param randf: 对参数的随机扰动函数，接受现有权值，返回扰动后的新权值np.array
        '''
        #初始化
        y_old = None
        while y_old == None or y_old < ybound[0] or y_old > ybound[1]:
            x_old = initf()
            y_old = f(x_old) 
        y_best = y_old
        x_best = np.copy(x_old)       
        #降温过程
        count = 0
        while(t > stop):
            downT = False
            for i in range(iterPerT):
                x_new = randf(x_old)
                y_new = f(x_new)    
                if y_new > ybound[1] or y_new < ybound[0]:
                    continue
                #根据取最大还是最小决定dE,最大为旧值尽可能小于新值
                dE = -(y_old - y_new) * l
                if dE < 0: 
                    downT = True
                    count = 0
                else: count += 1
                if self.__judge__(dE, t):
                    x_old = x_new
                    y_old = y_new
                    if y_old < y_best:
                        y_best = y_old
                        x_best = x_old
                    #绘图
                    if (count % 50 == 0):
                        plt.scatter(x_old[0], x_old[1])
                        plt.pause(0.001)
            if downT:
                t = t * alpha
            #长时间不降温
            if count > 1000: break
        self.weight = x_best
        return y_best
    
    def __judge__(self, dE, t):
        '''
        :param dE: 变化值\n
        :t: 温度\n
        根据退火概率: exp(-(E1-E2)/T)，决定是否接受新状态
        '''
        if dE < 0:
            return 1
        else:
            p = np.exp(-dE / t)
            import random
            
            if p > np.random.random(size=1):
                return 1
            else: return 0

def initRandom(constraint):
    #产生随机初始值，由列表表示权值的个数，每个权值的上下界由一个元组表示
    #ex.constraint = [(1,20),(13, 21)], 返回np.array
    weight = np.array([])
    for i in constraint:
        weight = np.append(weight, (i[1] - i[0]) * np.random.random() + i[0])
    return weight

def changeWeight(constraint, changeRange, now, bias=0):
    '''产生扰动后的权值，返回np.array
    :param constraint: 权值约束 :param changeRange: 权值扰动的范围
    :param bias: 权值扰动的偏好方向,eg: 0.1代表更倾向偏大的值20%
    :param now: 现有权值
    '''
    result = np.copy(now)
    for index in range(len(result)):
        delta = (np.random.random() - 0.5 + bias) * changeRange
        while (delta + result[index] > constraint[index][1] or
            delta + result[index] < constraint[index][0]):
            delta = (np.random.random() - 0.5 + bias) * changeRange
        result[index] += delta
    return result

def test1():
    #单变量非线性方程优化
    f = lambda x:x + 10 * np.sin(5 * x) + 7 * np.cos(4 * x)
    myConstraint = [(0, 9)] #约束
    init = lambda :initRandom(myConstraint) #随机初始化函数
    randf = lambda now:changeWeight(myConstraint, 3, now) #扰动函数

    temp = np.linspace(0, 10, 1000)
    plt.plot(temp, f(temp))
    sa = SAoptimizer()
    ans = sa.optimize(f, initf=init, randf=randf, l=1) #调用优化器
    plt.scatter(sa.weight, ans)
    print(sa.weight, ans)


def test2():
    #到点集的最短距离
    dots_x = [0, 1.5, 1.5, -2, -4, -5, 2, 4, 5, -2, -3 ,
        3, -1.5, -2.5, 0.1]
    dots_y = [0, 10, 12, 11 ,-8 ,2 ,-1.5 ,-2.5 ,1 ,-2 ,
        8 ,6 ,0 ,0 ,0.2]
    plt.scatter(dots_x, dots_y)
    #这里的评估指标，也就是质心到点集的距离需要通过f函数计算得出
    def f(weight):
        tot = 0
        for i in zip(dots_x, dots_y):
            tot += np.sqrt(
                (weight[0] - i[0]) ** 2 + (weight[1] - i[1]) ** 2)
        return tot
    myConstraint = [(-5, 5), (-8, 12)]
    init = lambda :initRandom(myConstraint)
    randf = lambda now: changeWeight(myConstraint, 1, now)
    
    sa = SAoptimizer()
    ans = sa.optimize(f, initf=init, randf=randf, stop=0.01, iterPerT=1)
    plt.scatter(sa.weight[0], sa.weight[1])
    print(sa.weight, ans)

def test3():
    #tsp问题
    cities = np.array([[0.9695,0.6606,0.5906,0.2124,0.0398,0.1367,0.9536,0.6091,0.8767,0.8148,0.3876,0.7041,0.0213,0.3429,0.7471,0.4606,0.7695,0.5006,0.3124,0.0098,0.3637,0.5336,0.2091,0.4767,0.4148,0.5876,0.6041,0.3213,0.6429,0.7471],
    [0.6740,0.9500,0.5029,0.8274,0.9697,0.5979,0.2184,0.7148,0.2395,0.2867,0.8200,0.3296,0.1649,0.3025,0.8192,0.6500,0.7420,0.0229,0.7274,0.4697,0.0979,0.2684,0.7948,0.4395,0.8867,0.3200,0.5296,0.3649,0.7025,0.9192]])
    plt.scatter(cities[0], cities[1])
    init = lambda :cities #参数为城市序列
    #这里需要自定义扰动函数
    def randf(now):
        import time
        new = np.copy(now)
        size = new.shape[1]
        while 1:
            city1 = np.random.randint(size)
            city2 = np.random.randint(size)
            if city1 != city2: break
        temp = np.copy(new[:, city1])
        new[:, city1] = new[:, city2]
        new[:, city2] = temp
        return new

    def f(weight):
        size = weight.shape[1]
        dist = 0
        for i in range(size - 1):
            dist += np.sqrt(np.sum(
                (weight[:, i + 1] - weight[:, i]) ** 2))
        # dist += np.sqrt(np.sum(
                # (weight[:, size - 1] - weight[:, 0]) ** 2))
        return dist

    sa = SAoptimizer()
    ans = sa.optimize(f, initf=init, randf=randf, stop=1e-5, t=1e3, alpha=0.99, l=10, iterPerT=1)
    for i in range(sa.weight.shape[1] - 1):
        plt.plot([sa.weight[0, i], sa.weight[0, i + 1]],
            [sa.weight[1, i], sa.weight[1, i + 1]])
    # plt.plot([sa.weight[0, 0], sa.weight[0, sa.weight.shape[1] - 1]],
        # [sa.weight[1, 0], sa.weight[1, sa.weight.shape[1] - 1]])
    print(ans)

if __name__ == '__main__':
    # test1()
    # test2()
    test3()
    pass