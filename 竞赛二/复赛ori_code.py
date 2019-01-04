import numpy as np
import pandas as pd
import tensorflow as tf
import string
import random
import random
from sklearn import tree
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import neural_network
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

#录入数据
def loadData(fileName):
    dataSet=[]
    with open(fileName) as txtData:
        lines=txtData.readlines()
        for line in lines:
            lineData=line.strip().split(',')
            dataSet.append(lineData)
    return dataSet
def testData(fileName):
    testSet = np.array(pd.read_csv(fileName,header=0))
    testSet = np.ndarray.tolist(testSet) 
    return testSet
#x,y分离
def splitData(dataSet):
    character_x=[]
    character_y=[]
    for i in range(len(dataSet)-1):
        character_x.append(dataSet[i][:-1])
        if dataSet[i][-1] == " <=50K":     #将y转化为0,1
            character_y.append(False)
        else:
            character_y.append(True)
    return np.array(character_x), np.array(character_y)
#字符串数字化
def numData(dataSet, labelName, labelNum):
    for i in range(len(dataSet)):
        for j in range(len(labelName)):
            if dataSet[i][labelNum] == labelName[j]:
                if dataSet[i][labelNum] == ' ?':  #空值整理
                    # dataSet[i][labelNum] = random.randint(0,len(labelName)-1)
                    dataSet[i][labelNum]=np.nan
                else:
                    dataSet[i][labelNum] = j
                break
            else:
                continue
    return dataSet
#整体空值整理
def spaceNum(dataSet, labelNum):
    for i in range(len(dataSet)-1):
        if dataSet[i][labelNum] == ' ?':
            # dataSet[i][labelNum] = dataSet[random.randint(0,len(dataSet)-1)][labelNum]
            dataSet[i][labelNum]=np.nan

    #下次体换成None试试
    return dataSet

if __name__ == "__main__":
    label=['age', 'workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country']
    dataset = loadData("./dataset/train-data.txt")
    testdata = testData("./dataset/testx.csv")
    dataset_x, dataset_y= splitData(dataset)
#-------------------字符串数字化---------------------
    #label[1]
    workclass=[' Private',' Self-emp-not-inc',' Self-emp-inc',' Federal-gov',' Local-gov',' State-gov',' Without-pay',' Never-worked',' ?']
    #label[3]
    education=[' Bachelors',' Some-college',' 11th',' HS-grad',' Prof-school',' Assoc-acdm',' Assoc-voc',' 9th',' 7th-8th',' 12th',' Masters',' 1st-4th',' 10th',' Doctorate',' 5th-6th',' Preschool']
    #label[5]
    marital_status=[' Married-civ-spouse',' Divorced',' Never-married',' Separated',' Widowed',' Married-spouse-absent',' Married-AF-spouse']
    #label[6]
    occupation=[' Tech-support',' Craft-repair',' Other-service',' Sales',' Exec-managerial',' Prof-specialty',' Handlers-cleaners',' Machine-op-inspct',' Adm-clerical',' Farming-fishing',' Transport-moving',' Priv-house-serv',' Protective-serv',' Armed-Forces',' ?']
    #label[7]
    relationship=[' Wife',' Own-child',' Husband',' Not-in-family',' Other-relative',' Unmarried']
    #label[8]
    race=[' White',' Asian-Pac-Islander',' Amer-Indian-Eskimo',' Other',' Black']
    #label[9]
    sex=[' Female',' Male'] 
    #label[13]
    native_country=[' United-States',' Cambodia',' England',' Puerto-Rico',' Canada',' Germany',' Outlying-US(Guam-USVI-etc)',' India',' Japan',' Greece',' South',' China',' Cuba',' Iran',' Honduras',' Philippines',' Italy',' Poland',' Jamaica', ' Vietnam',' Mexico',' Portugal',' Ireland',' France',' Dominican-Republic',' Laos',' Ecuador',' Taiwan',' Haiti',' Columbia',' Hungary', ' Guatemala',' Nicaragua',' Scotland',' Thailand',' Yugoslavia',' El-Salvador',' Trinadad&Tobago',' Peru',' Hong',' Holand-Netherlands',' ?']
   
    numData(dataset_x,workclass,1)
    numData(dataset_x,education,3)
    numData(dataset_x,marital_status,5)
    numData(dataset_x,occupation,6)
    numData(dataset_x,relationship,7)
    numData(dataset_x,race,8)
    numData(dataset_x,sex,9)
    numData(dataset_x,native_country,13)
    
    numData(testdata,workclass,1)
    numData(testdata,education,3)
    numData(testdata,marital_status,5)
    numData(testdata,occupation,6)
    numData(testdata,relationship,7)
    numData(testdata,race,8)
    numData(testdata,sex,9)
    numData(testdata,native_country,13)
#-------------------字符串数字化完成---------------------
#缺失处理    
    #整体空值
    for i in range(0,14):
        spaceNum(dataset_x,i)
        spaceNum(testdata,i)
#平均数填充
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(dataset_x)
    dataset_x = imp.transform(dataset_x)
    imp.fit(testdata)
    testdata = imp.transform(testdata)
#min-max标准化
    mms = preprocessing.MinMaxScaler()
    mms.fit(dataset_x)
    dataset_x = mms.transform(dataset_x)
    mms.fit(testdata)
    testdata = mms.transform(testdata)

    print("A beautiful day, right?")
    print("How is by you?")

    result =[]

#---------------------xgboost
    xgboost = xgb.XGBClassifier(nthread=4,learning_rate=0.28,\
                                n_estimators=50, max_depth=7, gamma=0.2, \
                                subsample=0.7, colsample_bytree=0.6)
    xgboost.fit(dataset_x, dataset_y)
    result = xgboost.predict(testdata)

#---------------------神经网络回归
    # mlp_reg = neural_network.MLPRegressor(hidden_layer_sizes=[64, 64], max_iter=100)
    # mlp_reg.fit(dataset_x, dataset_y)
    # result = mlp_reg.predict(testdata)

    resultE=[]

    for i in range(len(result)):
        if result[i]<0.5:
            resultE.append(bool(0))
        else:
            resultE.append(bool(1))

    numId=[]
    num = 0
    for i in range(len(resultE)):
        numId.append(i)

    resultE = np.array(resultE)

    print(resultE.dtype)
    dataframe = pd.DataFrame({'id':numId,'y':resultE})
    dataframe.to_csv("./dataset/testy.csv",index=False,sep=',')
    print(type(resultE[0]))
    print(resultE)

    print(dataset_x[14])