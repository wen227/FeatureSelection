# -*- coding: utf-8 -*-
# Feature Selection and Classification
# Importing the libraries
import scipy.io as sio
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, SelectFromModel, GenericUnivariateSelect, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

import numpy as np
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from joblib import dump, load
import sklearn_relief as relief
from sklearn.neighbors import KNeighborsClassifier


def multivariate_pearsonr(X, y):
    scores, pvalues = [], []
    for ret in map(lambda x: pearsonr(x, y), X.T):
        scores.append(abs(ret[0]))
        pvalues.append(ret[1])
    return (np.array(scores), np.array(pvalues))


def cm_result(cm):
    # Calculate the accuracy of a confusion_matrix,
    # where parameter 'cm' means confusion_matrix.
    a = cm.shape
    corrPred = 0
    falsePred = 0

    for row in range(a[0]):
        for c in range(a[1]):
            if row == c:
                corrPred += cm[row, c]
            else:
                falsePred += cm[row, c]
    Accuracy = corrPred / (cm.sum())
    return Accuracy


if __name__ == "__main__":
    # 存储特征信息
    dict_info = {}

    def select_features(model, X, name):
        # 根据训练完的模型进行特征选择
        # 返回各个特征选择方案的选择结果信息
        # X为待处理的特征矩阵
        data_ = model.transform(X)
        id_ = model.get_support(True)  # 选出的特征序号
        dict_info[name] = [data_, id_]

    # 载入数据
    file = sio.loadmat(r'data/SMK_CAN_187.mat')
    data = file['X']
    target = file['Y'][:, 0]
    # 数据预处理
    # 标准化
    data_std = StandardScaler().fit_transform(data)
    # 特征选择
    # 1.过滤式选择（Filter）
    # 1.1 单一特征选择(相关系数法)
    # 选择K个最好的特征，返回选择特征后的数据
    # 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。
    # 参数k为选择的特征个数
    model_pearsonr = SelectKBest(score_func=multivariate_pearsonr, k=20)
    model_pearsonr.fit(data_std, target)
    select_features(model_pearsonr, data_std, 'pearsonr')
    # data_pearsonr = SelectKBest(mutual_info_classif, k=20).fit_transform(data_std, target)
    # 1.2 单一特征选择(互信息法)
    model_mic = GenericUnivariateSelect(mutual_info_classif, 'k_best', param=20)
    model_mic.fit(data_std, target)
    select_features(model_mic, data_std, 'mic')
    # 1.3 多特征选择
    # r = relief.Relief(n_features=100, random_state=20, n_iterations=100)
    # model_Relief = r.fit(data_std, target)
    # dump(model_Relief, 'model_Relief.joblib')
    model_Relief = load('model_Relief.joblib')
    data_relief = model_Relief.transform(data_std)
    dict_info['Relief'] = [data_relief, 'NULL']

    # select_features(model_Relief, data_std, 'Relief')
    # 2.包裹式选择（Wrapper），本次实验采用递归特征消除法
    # 递归特征消除法，返回特征选择后的数据
    # 参数estimator为基模型
    # 参数n_features_to_select为选择的特征个数
    # 训练模型，测了三次，平均2小时
    # svcclassifier = LinearSVC(random_state=15)
    # print(svcclassifier)
    # model_wrapper = RFE(svcclassifier, n_features_to_select=20).fit(data_std, target)
    # dump(model_wrapper, 'model_wrapper.joblib')
    # data_rfe = model_wrapper.transform(data_std)
    # X_new = data_rfe
    # 直接加载训练完的模型
    model_ref = load('model_wrapper.joblib')
    data_rfe = model_ref.transform(data_std)
    # 打印出选出的特征序号
    temp = model_ref.ranking_
    top_k = -20
    top_k_idx = temp.argsort()[::-1][top_k:]
    dict_info['SVM-RFE'] = [data_rfe, top_k_idx]
    # 3.嵌入式选择（Embedded）
    clf = RandomForestClassifier(n_estimators=100, random_state=32)
    model_Embedded = SelectFromModel(clf)
    model_Embedded.fit(data_std, target)
    select_features(model_Embedded, data_std, 'Embedded')

    # 方案1：拆分成测试集和训练集
    # Splitting the dataset into the Training set and Test set
    # X_train, X_test, y_train, y_test = train_test_split(data_std, target, test_size=0.20, random_state=32)
    # svcclassifier = SVC(kernel='rbf', gamma='auto')
    # svcclassifier.fit(X_train, y_train)
    neigh = KNeighborsClassifier()
    # neigh.fit(X_train, y_train)

    # # Predicting the Train set results
    # y_pred_train = svcclassifier.predict(X_train)
    # y_pred_train = neigh.predict(X_train)
    # # y_pred_train = clf.predict(X_train)
    # # # Predicting the Test set results
    # y_pred = svcclassifier.predict(X_test)
    # y_pred = neigh.predict(X_test)
    # # y_pred = clf.predict(X_test)
    # # Making the Confusion Matrix
    # cm_test = confusion_matrix(y_test, y_pred)
    # cm_train = confusion_matrix(y_train, y_pred_train)
    # print(cm_train)
    # print(cm_test)
    # TrainAccuracy = cm_result(cm_train)
    # TestAccuracy = cm_result(cm_test)
    # y_compare = np.vstack((y_train, y_pred_train)).T
    # print('Accuracy of the train is: ', round(TrainAccuracy * 100, 2))
    # print('Accuracy of the test is: ', round(TestAccuracy * 100, 2))

    # 方案2：k折交叉验证
    # cross_val_predict
    for name in dict_info:
        print(name)
        target_pred = cross_val_predict(neigh, dict_info[name][0], target, cv=10)
        cm_kFold = confusion_matrix(target, target_pred)
        # finding accuracy from the confusion matrix.
        Accuracy = cm_result(cm_kFold)
        print('Accuracy of k-fold Clasification is: ', round(Accuracy * 100, 2))
        print(cm_kFold)
        print(dict_info[name][1])


