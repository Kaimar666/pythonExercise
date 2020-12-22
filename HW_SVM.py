#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
import sklearn as sl
import seaborn as sn
import matplotlib as mpl
from sklearn import svm
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit

train_data_url = r"C:/Users/81275/Desktop/UESTC/课程/研一上/机器学习/作业/第三次作业-SVM/data/diabetes_train.data"
test_data_url = r"C:/Users/81275/Desktop/UESTC/课程/研一上/机器学习/作业/第三次作业-SVM/data/diabetes_test.data"
#数据预处理
def load_data(train_url, test_url):
    train_data = np.genfromtxt(train_data_url, delimiter=',', comments='%', dtype='str')
    test_data = np.genfromtxt(test_data_url, delimiter=',', dtype='str')
    train_data[train_data == 'tested_negative'] = '0'
    train_data[train_data == 'tested_positive'] = '1'
    test_data[test_data == 'tested_negative'] = '0'
    test_data[test_data == 'tested_positive'] = '1'
    x_train = np.array(train_data[:,:-1].astype('float'))
    y_train = np.array(train_data[:,-1].astype('int'))
    x_test = np.array(test_data[:,:-1].astype('float'))
    y_test = np.array(test_data[:,-1].astype('int'))
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_data(train_data_url, test_data_url)
x = np.concatenate((x_train, x_test),axis=0)
y = np.concatenate((y_train, y_test),axis=0)
x = pd.DataFrame(x)
y = pd.DataFrame(y)
y.columns = ['Class']
sn.countplot(x = y['Class'])
plt.show()
# print(x)
# print(y)


# kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
# gamma_list = [5, 50, 250, 500]
# color = ['r', 'g', 'b', 'c']
# linestyle = ['-', '--', '-.', ':', '-.']
# marker = ['^', '*', '<', 'o']
#
# def plot_kernel(kernel, x_train, y_train, x, y, color, linestyle, marker):
#     svc = SVC(kernel = kernel)
#     svc.fit(x_train, y_train)
#     score_svm = svc.score(x_test,y_test)
#     print("SVM score: ",score_svm)
#
#     cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
#     svc_train_sizes, svc_train_score, svc_test_score = learning_curve(svc, x, y,
#                                                           train_sizes=np.linspace(0.1, 1, 10), cv=cv, n_jobs=1)
#     svc_test_score = svc_test_score.mean(axis=1)
#     plt.plot(svc_train_sizes, svc_test_score, color=color, linestyle=linestyle, marker=marker, label='kernel-'+kernel)
#
# # for i in range(4):
# #     plot_kernel(kernel_list[i], x_train, y_train, x, y, color[i], linestyle[i], marker[i])
# # plt.xlabel('train_sizes')
# # plt.ylabel('Accuracy')
# # plt.title('SVM-Kernel-Comparison')
# # plt.legend()
# # plt.show()
#
# def plot_gamma(gamma_list):
#     train_acc = []
#     test_acc = []
#     for i in gamma_list:
#         model = svm.SVC(kernel='rbf', gamma=i)
#         model.fit(x_train, y_train)
#         pred_score = model.predict(x_test)
#         pred_score = np.array(pred_score)
#
#         train_accuracy = model.score(x_train, y_train)
#         test_accuracy = model.score(x_test, y_test)
#         train_accuracy = round(train_accuracy, 2)
#         test_accuracy = round(test_accuracy, 2)
#         train_acc.append(train_accuracy)
#         test_acc.append(test_accuracy)
#
#     plt.figure()
#     plt.plot(gamma_list, train_acc, color='g', marker='^', label='train_acc')
#     plt.plot(gamma_list, test_acc, color='c', marker='*', label='test_acc')
#     plt.title('RBF kernel compare')
#     plt.xlabel('RBF gamma')
#     plt.ylabel('accuracy')
#     plt.legend(loc='center left')
#     plt.show()
#
# plot_gamma(gamma_list)








#搭建训练结构

#训练和测试
'''pred_score = model.predict(x_test)
pred_score = np.array(pred_score)

train_accuracy = model.score(x_train, y_train)
test_accuracy = model.score(x_test, y_test)
train_accuracy = round(train_accuracy, 2)
test_accuracy = round(test_accuracy, 2)'''

#可视化



'''fig = plt.figure()
plt.scatter(range(1, 195), pred_score, marker='*', color='c', label='predict')
plt.scatter(range(1, 195), y_test, marker='*', color='r', label='real')
plt.title('value of prediction and real')
plt.ylabel('value')
plt.xlabel('cases')
plt.legend(loc='center left')
plt.text(100, 0.6, F"train_accuracy:{train_accuracy}")
plt.text(100, 0.4, F"test_accuracy:{test_accuracy}")
plt.show()'''

