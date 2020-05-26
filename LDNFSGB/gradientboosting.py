'''
调用库函数
'''
import csv
import math
import random
import time
import torch
import pandas as pd  # 导入pandas包
from nn import np
from numpy import interp
from sklearn import metrics
from RotationForest import RotationForest
from DELM import HiddenLayer
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, confusion_matrix, auc, \
    precision_recall_curve
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def MyConfusionMatrix(y_real,y_predict):
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_real, y_predict)
    # print(CM)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    # print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
    Acc = (TN + TP) / (TN + TP + FN + FP)
    Sen = TP / (TP + FN)
    Spec = TN / (TN + FP)
    Prec = TP / (TP + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    Result = []
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))
    Result.append(round(MCC, 4))
    return Result
def MyAverage(matrix):
    SumAcc = 0
    SumSen = 0
    SumSpec = 0
    SumPrec = 0
    SumMcc = 0
    counter = 0
    while counter < len(matrix):
        SumAcc = SumAcc + matrix[counter][0]
        SumSen = SumSen + matrix[counter][1]
        SumSpec = SumSpec + matrix[counter][2]
        SumPrec = SumPrec + matrix[counter][3]
        SumMcc = SumMcc + matrix[counter][4]
        counter = counter + 1
    print('AverageAcc:',SumAcc / len(matrix))
    print('AverageSen:', SumSen / len(matrix))
    print('AverageSpec:', SumSpec / len(matrix))
    print('AveragePrec:', SumPrec / len(matrix))
    print('AverageMcc:', SumMcc / len(matrix))
    return

SampleFeature = []
feature=[]
ReadMyCsv(feature, "SampleFeatureAuto.csv")
for i in range(len(feature)):
    c = []
    for j in range(len(feature[0])):
        c.append(float(feature[i][j]))
    SampleFeature .append(c)
# print(len(SampleFeature))
# print(len(SampleFeature[0]))

# SampleLabel
SampleLabel = []
counter = 0
while counter < len(SampleFeature) / 2:
    SampleLabel.append(1)
    counter = counter + 1
counter1 = 0
while counter1 < len(SampleFeature) / 2:
    SampleLabel.append(0)
    counter1 = counter1 + 1

# 打乱数据集顺序
counter = 0
R = []
while counter < len(SampleFeature):
    R.append(counter)
    counter = counter + 1
random.shuffle(R)

RSampleFeature = []
RSampleLabel = []
counter = 0
while counter < len(SampleFeature):
    RSampleFeature.append(SampleFeature[R[counter]])
    RSampleLabel.append(SampleLabel[R[counter]])
    counter = counter + 1

print('len(RSampleFeature)', len(RSampleFeature))
print('len(RSampleLabel)', len(RSampleLabel))
SampleFeature = []
SampleLabel = []
SampleFeature = RSampleFeature
SampleLabel = RSampleLabel
X = np.array(SampleFeature)
y = np.array(SampleLabel)

if __name__ == '__main__':
    print("Start read data...")
    # n_estimators表示要组合的弱分类器个数；
    # algorithm可选{‘SAMME’, ‘SAMME.R’}，默认为‘SAMME.R’，表示使用的是real boosting算法，‘SAMME’表示使用的是discrete boosting算法
    seed=0
    num_tree = 1200
    SplitNum = 10
    cv = StratifiedKFold(n_splits=SplitNum)

    tprs1 = []
    aucs1 = []
    AverageResult1 = []
    mean_fpr1 = np.linspace(0, 1, 100)

    tprs2 = []
    aucs2 = []
    AverageResult2 = []
    mean_fpr2 = np.linspace(0, 1, 100)

    tprs3 = []
    aucs3 = []
    AverageResult3 = []
    mean_fpr3 = np.linspace(0, 1, 100)

    tprs4 = []
    aucs4 = []
    AverageResult4 = []
    mean_fpr4 = np.linspace(0, 1, 100)

    tprs5 = []
    aucs5 = []
    AverageResult5 = []
    mean_fpr5 = np.linspace(0, 1, 100)

    tprs6 = []
    aucs6 = []
    AverageResult6 = []
    mean_fpr6 = np.linspace(0, 1, 100)

    tprs7 = []
    aucs7 = []
    AverageResult7 = []
    mean_fpr7 = np.linspace(0, 1, 100)

    tprs8 = []
    aucs8 = []
    AverageResult8 = []
    mean_fpr8 = np.linspace(0, 1, 100)

    for train, test in cv.split(X, y):
        model1 = SVC(probability=True)
        model1.fit(X[train], y[train])
        y_score01 = model1.predict(X[test])
        y_score1 = model1.predict_proba(X[test])
        Result1 = MyConfusionMatrix(y[test], y_score01)  #
        AverageResult1.append(Result1)
        fpr1, tpr1, thresholds1 = roc_curve(y[test], y_score1[:, 1])
        tprs1.append(interp(mean_fpr1, fpr1, tpr1))
        tprs1[-1][0] = 0.0
        roc_auc1 = auc(fpr1, tpr1)
        aucs1.append(roc_auc1)
    # 画均值
    print('#------------------SVM----------------------#')
    MyAverage(AverageResult1)
    mean_tpr1 = np.mean(tprs1, axis=0)
    mean_tpr1[-1] = 1.0
    mean_auc1 = auc(mean_fpr1, mean_tpr1)
    std_auc1 = np.std(aucs1)
    plt.plot(mean_fpr1, mean_tpr1, label=r'SVM (AUC = %0.4f)' % (mean_auc1),
             lw=1, alpha=.8, color='#FF69B4')

    for train, test in cv.split(X, y):
        model2 = GaussianNB()
        model2.fit(X[train], y[train])
        y_score02 = model2.predict(X[test])
        y_score2 = model2.predict_proba(X[test])
        Result2 = MyConfusionMatrix(y[test], y_score02)  #
        AverageResult2.append(Result2)
        fpr2, tpr2, thresholds2 = roc_curve(y[test], y_score2[:, 1])
        tprs2.append(interp(mean_fpr2, fpr2, tpr2))
        tprs2[-1][0] = 0.0
        roc_auc2 = auc(fpr2, tpr2)
        aucs2.append(roc_auc2)
    # 画均值
    print('#------------------NaiveBayes----------------------#')
    MyAverage(AverageResult2)
    mean_tpr2 = np.mean(tprs2, axis=0)
    mean_tpr2[-1] = 1.0
    mean_auc2 = auc(mean_fpr2, mean_tpr2)
    std_auc2 = np.std(aucs2)
    plt.plot(mean_fpr2, mean_tpr2, label=r'NaiveBayes (AUC = %0.4f)' % (mean_auc2),
             lw=1, alpha=.8, color='#EE82EE')

    for train, test in cv.split(X, y):
        model3 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        model3.fit(X[train], y[train])
        y_score03 = model3.predict(X[test])
        y_score3 = model3.predict_proba(X[test])
        Result3 = MyConfusionMatrix(y[test], y_score03)  #
        AverageResult3.append(Result3)
        fpr3, tpr3, thresholds3 = roc_curve(y[test], y_score3[:, 1])
        tprs3.append(interp(mean_fpr3, fpr3, tpr3))
        tprs3[-1][0] = 0.0
        roc_auc3 = auc(fpr3, tpr3)
        aucs3.append(roc_auc3)
    # 画均值
    print('#------------------LogisticRegression----------------------#')
    MyAverage(AverageResult3)
    mean_tpr3 = np.mean(tprs3, axis=0)
    mean_tpr3[-1] = 1.0
    mean_auc3 = auc(mean_fpr3, mean_tpr3)
    std_auc3 = np.std(aucs3)
    plt.plot(mean_fpr3, mean_tpr3, label=r'LogisticRegression (AUC = %0.4f)' % (mean_auc3),
             lw=1, alpha=.8, color='#7B68EE')

    for train, test in cv.split(X, y):
        model4 = RandomForestClassifier()
        model4.fit(X[train], y[train])
        y_score04 = model4.predict(X[test])
        y_score4 = model4.predict_proba(X[test])
        Result4 = MyConfusionMatrix(y[test], y_score04)  #
        AverageResult4.append(Result4)
        fpr4, tpr4, thresholds4 = roc_curve(y[test], y_score4[:, 1])
        tprs4.append(interp(mean_fpr4, fpr4, tpr4))
        tprs4[-1][0] = 0.0
        roc_auc4 = auc(fpr4, tpr4)
        aucs4.append(roc_auc4)
    # 画均值
    print('#------------------RandomForest----------------------#')
    MyAverage(AverageResult4)
    mean_tpr4 = np.mean(tprs4, axis=0)
    mean_tpr4[-1] = 1.0
    mean_auc4 = auc(mean_fpr4, mean_tpr4)
    std_auc4 = np.std(aucs4)
    plt.plot(mean_fpr4, mean_tpr4, label=r'RandomForest (AUC = %0.4f)' % (mean_auc4),
             lw=1, alpha=.8, color='#6495ED')

    for train, test in cv.split(X, y):
        model5 = RotationForest()
        model5.fit(X[train], y[train])
        y_score05 = model5.predict(X[test])
        y_score5 = model5.predict_proba(X[test])
        Result5 = MyConfusionMatrix(y[test], y_score05)  #
        AverageResult5.append(Result5)
        fpr5, tpr5, thresholds5 = roc_curve(y[test], y_score5[:, 1])
        tprs5.append(interp(mean_fpr5, fpr5, tpr5))
        tprs5[-1][0] = 0.0
        roc_auc5 = auc(fpr5, tpr5)
        aucs5.append(roc_auc5)
    # 画均值
    print('#------------------RotationForest----------------------#')
    MyAverage(AverageResult5)
    mean_tpr5 = np.mean(tprs5, axis=0)
    mean_tpr5[-1] = 1.0
    mean_auc5 = auc(mean_fpr5, mean_tpr5)
    std_auc5 = np.std(aucs5)
    plt.plot(mean_fpr5, mean_tpr5, label=r'RotationForest (AUC = %0.4f)' % (mean_auc5),
             lw=1, alpha=.8, color='#00FFFF')

    for train, test in cv.split(X, y):
        model6 = AdaBoostClassifier(n_estimators=num_tree, learning_rate=1,random_state=seed)
        model6.fit(X[train], y[train])
        y_score06 = model6.predict(X[test])
        y_score6 = model6.predict_proba(X[test])
        Result6 = MyConfusionMatrix(y[test], y_score06)  #
        AverageResult6.append(Result6)
        fpr6, tpr6, thresholds6= roc_curve(y[test], y_score6[:, 1])
        tprs6.append(interp(mean_fpr6, fpr6, tpr6))
        tprs6[-1][0] = 0.0
        roc_auc6 = auc(fpr6, tpr6)
        aucs6.append(roc_auc6)
    # 画均值
    print('#------------------AdaBoost----------------------#')
    MyAverage(AverageResult6)
    mean_tpr6 = np.mean(tprs6, axis=0)
    mean_tpr6[-1] = 1.0
    mean_auc6 = auc(mean_fpr6, mean_tpr6)
    std_auc6 = np.std(aucs6)
    plt.plot(mean_fpr6, mean_tpr6, label=r'AdaBoost (AUC = %0.4f)' % (mean_auc6),
             lw=1, alpha=.8, color='#98FB98')

    for train, test in cv.split(X, y):
        model8 = HiddenLayer(X[train], 490)
        model8.classifisor_train(y[train])
        result8, pro8 = model8.classifisor_test(X[test])
        output8 = F.softmax(torch.from_numpy(np.array(pro8)), dim=1)
        # print(metrics.accuracy_score(y[test], result8))
        res_1 = []
        for j in range(len(output8)):
            res_1.append(output8[j][1])
        fpr8, tpr8, threshold8 = roc_curve(y[test], res_1)  ###计算真正率和假正率
        Result8 = MyConfusionMatrix(y[test], result8)  #
        AverageResult8.append(Result8)
        tprs8.append(interp(mean_fpr8, fpr8, tpr8))
        tprs8[-1][0] = 0.0
        roc_auc8 = auc(fpr8, tpr8)
        aucs8.append(roc_auc8)

    print('#------------------DELM----------------------#')
    MyAverage(AverageResult8)
    mean_tpr8 = np.mean(tprs8, axis=0)
    mean_tpr8[-1] = 1.0
    mean_auc8 = auc(mean_fpr8, mean_tpr8)
    std_auc8 = np.std(aucs8)
    plt.plot(mean_fpr8, mean_tpr8, label=r'DELM (AUC = %0.4f)' % (mean_auc8),
             lw=1, alpha=.8, color='#DAA520')

    for train, test in cv.split(X, y):
        model7 = GradientBoostingClassifier(n_estimators=num_tree, random_state=seed)
        model7.fit(X[train], y[train])
        y_score07 = model7.predict(X[test])
        y_score7 = model7.predict_proba(X[test])
        Result7 = MyConfusionMatrix(y[test], y_score07)  #
        AverageResult7.append(Result7)
        fpr7, tpr7, thresholds7= roc_curve(y[test], y_score7[:, 1])
        tprs7.append(interp(mean_fpr7, fpr7, tpr7))
        tprs7[-1][0] = 0.0
        roc_auc7 = auc(fpr7, tpr7)
        aucs7.append(roc_auc7)
    # 画均值
    print('#------------------GradientBoosting----------------------#')
    MyAverage(AverageResult7)
    mean_tpr7 = np.mean(tprs7, axis=0)
    mean_tpr7[-1] = 1.0
    mean_auc7 = auc(mean_fpr7, mean_tpr7)
    std_auc7 = np.std(aucs7)
    plt.plot(mean_fpr7, mean_tpr7, label=r'GradientBoosting(AUC = %0.4f)' % (mean_auc7),
             lw=1, alpha=.8, color='#FF0000')
    # 画标题坐标轴
    plt.xlabel('FPR', fontsize=13)
    plt.ylabel('TPR', fontsize=13)
    # plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('comparapre (AUC = %0.4f).png'% (mean_auc7), dpi=300)
    plt.show()


    plt.figure(2)
    plt.plot(mean_fpr7, mean_tpr7, label=r'ROC 10-fold CV(AUC = %0.4f)' % (mean_auc7),
             lw=1, alpha=.8, color='#FF0000')
    # 画标题坐标轴
    plt.xlabel('FPR', fontsize=13)
    plt.ylabel('TPR', fontsize=13)
    # plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC 10-fold CV (AUC = %0.4f).png' % (mean_auc7), dpi=300)
    plt.show()





