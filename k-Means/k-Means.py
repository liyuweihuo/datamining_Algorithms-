import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataList = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float , curLine))
            dataList.append(fltLine)
    return dataList
def randCent(dataSet , k):#随机生成k个初始质心
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = np.mat(minJ + rangeJ*np.random.rand(k,1))
    return centroids
def k_Means(dataSet , k):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids = randCent(dataSet , k)
    clusterChanged = True
    iterIndex = 1
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf;minIndex = -1
            for j in range(k):
                distJI = np.linalg.norm(np.array(centroids[j,:])-np.array(dataSet[i,:]))
                if distJI < minDist:
                    minDist = distJI ; minIndex= j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
            print("第%d次迭代后%d个质心的坐标:\n%s"%(iterIndex,k,centroids))
            iterIndex +=1
        for cent in range(k):#重新计算质心，是现cluster的均值
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust , axis = 0)
    return centroids , clusterAssment
def showCluster(dataSet , k ,centroids,clusterAssment):
    numSamples , dim = dataSet.shape
    if dim!=2:
        return 1
    mark = ['or' , 'ob' ,'og' , 'ok' , 'oy' , '^r' , '+r' , 'sr' , 'dr'  ,'<r' , 'pr']
    for i in range(numSamples):
        markIndex = int(clusterAssment[i,0])
        plt.plot(dataSet[i,0],dataSet[i,1],mark[markIndex])
    mark = ['Pr' , 'Pb' , 'Pg' , 'Pk' , 'Py' , 'Pm' , 'Pc' , '^b' , '+b' , 'sb' , 'db' , '<b' , 'pb']
    for i in range(k):
        plt.plot(centroids[i,0],centroids[i,1],mark[i] , markersize = 12)
    plt.show()
if __name__ == '__main__':
    dataMat = np.mat(loadDataSet('E:\\data\\datamining_algorithms-master\\K-means\\testSet'))
    k = 4
    cent , clust = k_Means(dataMat , k)

    showCluster(dataMat , k ,cent,clust)