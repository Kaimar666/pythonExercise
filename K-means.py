import pandas as pd
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
from math import *
from numpy import *

path = r"C:/Users/81275/Desktop/UESTC/课程/研一上/机器学习/作业/第五次作业-聚类/第八章作业/数据集/K-means.data"

# 数据读取
def dataProcessing(path):
    data = pd.read_csv(path)
    data = np.array(data)
    # print(data)
    # plt.scatter(x=data['x'], y=data['y'])
    # plt.show()
    return data

# 计算欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # 求两个向量之间的距离(此公式即两点之间的距离公式)

# 构建簇中心
def randCent(dataSet, k):
    n = shape(dataSet)[1]  # 数据集是(221,2) 此处n=2
    centroids = mat(zeros((k,n)))   # 其中zeros((k,n))生成k*n的全零填充array. 每个质心有n个坐标值,总共要k个质心.(此变量是存储k个中心点对应的x,y坐标)
    for j in range(n):
        minJ = min(dataSet[:,j])  # 得到最大、最小的x和y值
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)  # 得到x和y的范围
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1) # rand生成一个k*1的array,并且值在[0,1)
    return centroids

# K-means算法
def kMeans(dataset, k, distMeans = distEclud, createCent = randCent):
    m = shape(dataset)[0]  # (221, 2)  此处m=221
    clusterAssment = mat(zeros((m, 2)))  # 第一列存放该数据所属的中心点,第二列是该数据到中心点的距离(此数组记录每个点属于哪个类)
    centroids = createCent(dataset, k)  # 创建簇中心  centroids是一个4*2的array,对应4个质心的坐标
    clusterChanged = True  # 判断聚类收敛的标识
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf  # inf指正无穷,nan指负无穷
            minIndex = -1  # 初始化最小距离为无穷大,对应
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataset[i,:])  # 质心的每一行与数据集的每一行求距离(即求每个点到哪个质心的距离最小)
                if distJI < minDist:  # 找到更小,则记录其距离和所属质心(如果第i个数据点到第j个中心点更近，则将i归属为j)
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:  # 若对应的质心不同,则更新
                clusterChanged = True            # 若发生更新,则说明质心有变化,还需继续迭代,直到质心无变化为止
                clusterAssment[i,:] = minIndex, minDist**2  # 由于计算距离的公式最后开了二次根,所以对应距离应该是要平方一下
        print(centroids)
        # 计算每个簇的均值,修改质心
        for cent in range(k):
            ptsInClust = dataset[nonzero(clusterAssment[:,0].A == cent)[0]]  # 1、clusterAssment[:,0]得到的是数据集中每个点对应的质心(数值分别为0,1,2,3)
            centroids[cent,:] = mean(ptsInClust, axis=0)                     # 2、.A==cent是将上述对应数值不等于cent的返回False,相等为True,得到一个同上结构的bool类型数组
    return centroids, clusterAssment                                         # 3、nonzero返回数组中非零元素对应的索引,最后ptsInClust得到的是属于同一质心的所有数据集中的点的坐标

if __name__ == "__main__":
    dataset = dataProcessing(path)
    m = shape(dataset)[0]
    # centroids存质心的坐标, clusterAssment存质心和距质心的距离
    myCentroids, myClusterAssment = kMeans(dataset, 4)
    print('----------------------------------------------------------------')
    print(myCentroids)
    print(myClusterAssment)
    colors = ['r', 'y', 'b', 'g']
    for i in range(4):
        plt.scatter(myCentroids[i,0], myCentroids[i,1], color=colors[i], marker='*')
    for j in range(m):
        plt.scatter(dataset[j,0], dataset[j,1], color=colors[int(myClusterAssment[j,0])])
    plt.show()