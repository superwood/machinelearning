#coding=utf-8
__author__ = 'superwood'
from numpy import *
import numpy as np
from math import *
import operator
import matplotlib
import matplotlib.pyplot as plt


def loadDataSet(filename):
    dataMat = []; labelMat = []
    fp = open(filename)
    linenumber = 0
    for line in fp:
        if linenumber == 0:  ##跳过头部schema
            linenumber  = linenumber + 1
            continue
        lineArr = line.strip().split("\n")[0].split(",")
        datalineX = []
        for index in range(7,9): ## 数据需要的三列
            datalineX.append( eval( lineArr[index]) )
        datalineX.append(1.0) #(X:b),后面追加一个b项的权重值
        dataMat.append(datalineX)
        if lineArr[9] == '是':
            labelMat.append(int(1))
        else:
            labelMat.append(int(0))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/( 1 + exp(-inX) )

def newtonMethod(dataMat, labelMat):
    dataMatrix = mat(dataMat)
    labelMatrix = mat(labelMat)

    sampleNum,dimNum = shape(dataMat) ##样本个数, 特征维度
    paramTheta = np.mat( np.zeros(dimNum) ) ##初始化参数(W:b)
    cLikelihood = 0
    oLikelihood = 1

    threshold = 0.0001
    while True:
        if math.fabs(cLikelihood - oLikelihood) < threshold:
            break

        oLikelihood = cLikelihood
        iteraTheta,cLikelihood = computeDers(dataMatrix,labelMatrix, paramTheta)
        paramTheta = paramTheta - iteraTheta
        print oLikelihood,cLikelihood,paramTheta
    return paramTheta

def computeDers(dataMatrix, labelMatrix, param):
    """参数  dataMat 数据  labelMat 标签  param当前的beta参数
       返回  iterator value for w(param)   and current likehoold
    """
    sampleNum,dimNum = shape(dataMatrix) ##样本个数, 特征维度
    derOne = [0.0]*dimNum

    derTwo = mat(np.zeros((3,3) ) )
    curLikelihood = 0
    for i in range(sampleNum):
        paramX = param*dataMatrix[i].T  ## W*T
        expX   = exp(paramX.getA()[0][0] )  ##  e^{wt}
        Y = labelMatrix.getA()[0][i]
        derOne = derOne - dataMatrix[i]*Y + dataMatrix[i]*(expX)/(1+expX) ##一阶导数
        derTwo = derTwo + dataMatrix[i].T*dataMatrix[i]*expX/( (1+expX)*(1+expX)) ## 二阶导数
        curLikelihood = curLikelihood - Y*paramX + math.log(1+expX)
    tmp = derOne * derTwo.I ##iterator value for w
    return tmp,curLikelihood.getA()[0][0]
def error(param, example, result):
    paramX = np.dot(param, example)
    y = 1/(1+math.exp(-paramX))

    if y > 0.5:
        z = 1
    else:
        z = 0
    print paramX, y, z,result
    if result == z:
        return 0
    else:
        return 1
def errorRate(filename,param):
    file = open(filename)
    index = 0
    total = 0;
    errornum = 0
    for line in file:
        dataliney=[]
        datalinex=[]
        if index == 0:
            index += 1
            continue
        listLine = line.split("\n")[0].split(",")
        for index in range(7,9):
            datalinex.append( eval(listLine[index]) )
        datalinex.append(1)
        if listLine[9] == "是":
            dataliney.append(1)
        else:
            dataliney.append(0)
        errornum += error(param, datalinex, dataliney[0])
        total += 1
    print errornum, total
    return errornum*1.0/total
def plotBestFit(dataMat, labelMat, weights):
    """"""
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,0]); ycord1.append(dataArr[i,1])
        else:
            xcord2.append(dataArr[i,0]); ycord2.append(dataArr[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='+')
    ax.scatter(xcord2, ycord2, s=30, c='green', marker='o')
    x = arange(0.2, 2, 0.5)
    y = (-weights[2]-weights[0]*x)/weights[1]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


if __name__ == '__main__':
    ##param = data2matrix("../data/watermelon.txt")
    ##print param
    dataMatrix,lableMatrix = loadDataSet("../data/watermelon.txt")
    paramTheta = newtonMethod(dataMatrix,lableMatrix)
    plotBestFit(dataMatrix, lableMatrix, paramTheta.getA()[0])
    #errorRate("../data/watermelon.txt", param)
