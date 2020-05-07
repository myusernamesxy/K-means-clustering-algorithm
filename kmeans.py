import numpy as np
import random
import matplotlib.pyplot as plt


def loadData(filename):
    valuedata=[]
    tagdata=[]
    with open(filename, 'r') as fn:
        data = fn.readlines()
        for line in data:
            line = line.split(',')
            lines = []
            for i in range(len(line) - 1):
                lines.append(float(line[i]))
            label = line[len(line) - 1].strip("\n")
            if label == 'Iris-setosa':
                tagdata.append(1)
            elif label == 'Iris-versicolor':
                tagdata.append(2)
            else:
                tagdata.append(3)
            valuedata.append(lines)
    return valuedata, tagdata


def initial(valuedata, k):
    index = random.sample(range(0, len(valuedata)), k)
    centroid = []
    for i in range(k):
        centroid.append(valuedata[index[i]])
    return centroid


def distance(data1, data2):
    d = 0
    for i in range(len(data1)):
        d += pow(data1[i] - data2[i], 2)
    return pow(d, .5)


def kmeans(valuedata, k):
    centroid = initial(valuedata, k)
    diff = 1.0
    # update the centroids
    while diff > 0.0:
        sumcluster = np.zeros((k, 4))
        cluster = np.zeros((len(valuedata), k))
        count = [0] * k
        for j in range(len(valuedata)):
            mindistance = distance(centroid[0], valuedata[j])
            index = 0
            for ind in range(1, k):
                d = distance(centroid[ind], valuedata[j])
                if d < mindistance:
                    mindistance = d
                    index = ind
            cluster[count[index], index] = j
            sumcluster[index] += valuedata[j]
            count[index] += 1
        diff = 0.0
        tmp = np.zeros((1, 4))
        for i in range(k):
            tmp += abs(sumcluster[i]/count[i] - centroid[i])
            # calculate the mean values as new centroids
            centroid[i] = sumcluster[i]/count[i]
        for num in range(4):
            diff += tmp[0, num]
    return cluster, count, centroid


def getmse(valuedata, k):
    times = 0
    min = 2147483647
    # executed the K-means algorithm multiple times to avoid local optimum
    while times < 5:
        cluster, count, centroid = kmeans(valuedata, k)
        res = 0
        for cen in range(k):
            for index in range(count[cen]):
                # print(valuedata[cluster[index, cen]])
                res += distance(valuedata[int(cluster[index, cen])], centroid[cen])
        # store the best results
        if res < min:
            min = res
            newcluster = cluster
            newcount = count
            newcentroid = centroid
        times += 1
    for num in range(k):
        cls = []
        for i in range(newcount[num]):
            cls.append(valuedata[int(newcluster[i][num])])
        print('cluster', num, 'is: ', cls)
    return min, newcluster, newcount, newcentroid


def accuracy(cluster, count):
    correct = 0
    for kth in range(3):
        cls3 = np.zeros((1, 3))
        for num in range(count[kth]):
            if cluster[num][kth] < 50:
                cls3[0][0] += 1
            elif cluster[num][kth] < 100:
                cls3[0][1] += 1
            else:
                cls3[0][2] += 1
        if cls3[0][0] > cls3[0][1]:
            if cls3[0][0] > cls3[0][2]:
                correct += cls3[0][0]
            else:
                correct += cls3[0][2]
        else:
            if cls3[0][1] > cls3[0][2]:
                correct += cls3[0][1]
            else:
                correct += cls3[0][2]
    return correct/150


def showCluster(valuedata, k, centroids, cluster, count):
    mark = ['or', 'ob', 'og', 'ok']
    markcen = ['Dr', 'Db', 'Dg', 'Dk']
    for i in range(k):
        plt.plot(centroids[i][0], centroids[i][1], markcen[i], markersize=9)
        for m in range(count[i]):
            plt.plot(valuedata[int(cluster[m][i])][0], valuedata[int(cluster[m][i])][1], mark[i])
    plt.show()


if __name__ == "__main__":
    valuedata, tagdata = loadData('data.txt')
    min, cluster, count, centroid = getmse(valuedata, 3)
    showCluster(valuedata, 3, centroid, cluster, count)
    print('The sum of squared distance differences is: ', min)
    print('The accuracy is(according to the data labels, only K=3): ', accuracy(cluster, count))
