# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 10:41:35 2018

@author: QM-GC
"""

import numpy as np
import pandas as pd

def createData():
    filename1 = "./dataset/trainx.csv"
    filename2 = "./dataset/trainy.csv"
    dataSet = []
    trainx = np.array(pd.read_csv(filename1,header=0))
    trainy = np.array(pd.read_csv(filename2,header=0))
    dataSet = np.ndarray.tolist(np.hstack((trainx, trainy)))
    print(type(dataSet))
    return dataSet

def testData():
    filename3 = "./dataset/testx.csv"
    testSet = np.array(pd.read_csv(filename3,header=0))
    testSet = np.ndarray.tolist(testSet) 
    return testSet


# 计算总的方差
def GetAllVar(dataSet):
    return np.var(dataSet[:,-1])*np.shape(dataSet)[0]   #var()方差，shape()纬度

# 根据给定的特征、特征值划分数据集
def dataSplit(dataSet,feature,featNumber):
    dataL =  dataSet[np.nonzero(dataSet[:,feature] > featNumber)[0],:]  #布尔转为整数
    dataR = dataSet[np.nonzero(dataSet[:,feature] <= featNumber)[0],:]
    return dataL,dataR

# 特征划分
def choseBestFeature(dataSet,op = [1,4]):          # 预剪枝操作
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:     # 标签相同
        regLeaf = np.mean(dataSet[:,-1])         #以剩下的均值为最佳划分值
        return None,regLeaf                   # 返回标签的均值作为叶子节点
    Serror = GetAllVar(dataSet)
    BestFeature = -1; BestNumber = 0; lowError = np.inf #方差 无限大
    m,n = np.shape(dataSet) # m 个样本， n -1 个特征
    for i in range(n-1):    # 遍历每一个特征值
        for j in set(dataSet[:,i].T.tolist()[0]): #tolist转成列表
            dataL,dataR = dataSplit(dataSet,i,j)
            if np.shape(dataR)[0]<op[1] or np.shape(dataL)[0]<op[1]: # 划分后样本数目少，跳出
                continue  
            tempError = GetAllVar(dataL) + GetAllVar(dataR)
            if tempError < lowError:  #当前方差小，就替换成新的特征
                lowError = tempError; BestFeature = i; BestNumber = j
    if Serror - lowError < op[0]:               # 总方差差距很小，数据不划分，返回均值
        return None,np.mean(dataSet[:,-1])         
    dataL, dataR = dataSplit(dataSet, BestFeature, BestNumber)
    if np.shape(dataR)[0] < op[1] or np.shape(dataL)[0] < op[1]:        # 划分后样本数量较小，返回均值
        return None, np.mean(dataSet[:, -1])
    return BestFeature,BestNumber


# 决策树生成
def createTree(dataSet,op=[1,4]):
    bestFeat,bestNumber = choseBestFeature(dataSet,op)
    if bestFeat==None: #到最后，没有属性，只剩叶节点
        return bestNumber
    regTree = {}
    regTree['spInd'] = bestFeat
    regTree['spVal'] = bestNumber
    dataL,dataR = dataSplit(dataSet,bestFeat,bestNumber)  #正式处理处理过的数据
    regTree['left'] = createTree(dataL,op)
    regTree['right'] = createTree(dataR,op)
    return  regTree

# 后剪枝操作
def isTree(Tree):   #是否是叶子节点
    return (type(Tree).__name__=='dict' )

# 计算两个叶子节点的均值
def getMean(Tree):
    if isTree(Tree['left']):
        Tree['left'] = getMean(Tree['left'])  #如果是叶子节点，就将剩下的样本取均值作为节点
    if isTree(Tree['right']):
        Tree['right'] = getMean(Tree['right'])
    return (Tree['left']+ Tree['right'])/2.0

def prune(Tree,testData):
    if np.shape(testData)[0]==0:  #叶子节点，取剩下的均值
        return getMean(Tree) 
    if isTree(Tree['left'])or isTree(Tree['right']): #不是叶子节点，就按照tree进行测试集划分
        dataL,dataR = dataSplit(testData,Tree['spInd'],Tree['spVal'])
    #划分当前后，迭代划分子树
    if isTree(Tree['left']): 
        Tree['left'] = prune(Tree['left'],dataL)
    if isTree(Tree['right']):
        Tree['right'] = prune(Tree['right'],dataR)
    #左右都划分完了
    if not isTree(Tree['left']) and not isTree(Tree['right']):
        dataL,dataR = dataSplit(testData,Tree['spInd'],Tree['spVal']) #处理叶子节点
        errorNoMerge = sum(np.power(dataL[:,-1] - Tree['left'],2)) + sum(np.power(dataR[:,-1] - Tree['right'],2))
        leafMean = getMean(Tree)
        errorMerge = sum(np.power(testData[:,-1]-  leafMean,2))
        if errorNoMerge > errorMerge:  #以方差作为评判标准，哪个效果好返回哪个
            print("the leaf merge")
            return leafMean
        else:
            return Tree
    else:
        return Tree
    


# 预测
def forecastSample(Tree,testData):  #根据训练集，找到相应的叶子节点，输出0-1之间的可能值
    if not isTree(Tree): 
        return float(Tree)
    if testData[0,Tree['spInd']]>Tree['spVal']:
        if isTree(Tree['left']):
            return forecastSample(Tree['left'],testData)
        else:
            return float(Tree['left'])
    else:   
        if isTree(Tree['right']):
            return forecastSample(Tree['right'],testData)
        else:
            return float(Tree['right'])

def TreeForecast(Tree,testData):
    m = np.shape(testData)[0]
    y_hat = np.mat(np.zeros((m,1))) #存储结果
    for i in range(m): #遍历所有值预测结果
        y_hat[i,0] = forecastSample(Tree,testData[i])
    return y_hat



if __name__=="__main__":
    print ("hello world")
    dataMat = createData()
    dataMat = np.mat(dataMat)
    op = [1,6]    # 方差最小值，样本最少数        
    theCreateTree =  createTree(dataMat,op)
   # 测试数据
    dataMat2 = testData()
    dataMat2 = np.mat(dataMat2)

    y = dataMat2[:, -1]
    y_hat = TreeForecast(theCreateTree,dataMat2)
    print("end")
    print(np.corrcoef(y_hat,y,rowvar=0)[0,1])        # 用预测值与真实值计算相关系数

    
    result=[]
    
    for i in range(0,2000):
        if y_hat[i]<0.5:  #将结果四舍五入
            result.append(0)
        else:
            result.append(1)  
    
    df = pd.DataFrame(result)
    df.to_csv(r"./dataset/testy.csv",index=False,encoding= u'utf-8')




    

