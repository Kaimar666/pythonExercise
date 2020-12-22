import os
import numpy as np
import tensorflow as tf
from time import time
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import label_binarize, OneHotEncoder
from sklearn.metrics import roc_curve, auc

#tools
def plot_roc(name, num, n_classes, Ytest, y_score):
    # roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
    # 横坐标：假正率（False Positive Rate , FPR）
    # Compute ROC curve and ROC area for each class
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Ytest[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves

    lw = 2
    #plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red']
    linestyles = [':', '--', '-.', '-']
    plt.plot(fpr["macro"], tpr["macro"],
             label=name+'-'+'macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]), color=colors[num], linestyle=linestyles[num], linewidth=4)

t = time()

x_train_savepath = r"C:/Users/81275/Desktop/UESTC/datasets/mnist/x_train.npy"
y_train_savepath = r"C:/Users/81275/Desktop/UESTC/datasets/mnist/y_train.npy"
x_test_savepath = r"C:/Users/81275/Desktop/UESTC/datasets/mnist/x_test.npy"
y_test_savepath = r"C:/Users/81275/Desktop/UESTC/datasets/mnist/y_test.npy"
#load_data
if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(x_test_savepath) and os.path.exists(y_test_savepath):
    print('---------------loadDatasets---------------')
    x_train = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    # x_train = np.reshape(x_train, (len(x_train), 28, 28))
    # x_test = np.reshape(x_test, (len(x_test), 28, 28))
else:
    print('!!!!!!!!!!!!!!!!!!No Datasets!!!!!!!!!!!!!!!!!')

#data_preprocessing
y_train[y_train != 0] = 1
y_test[y_test != 0] = 1
y_test_b = to_categorical(y_test)
n_classes = 2

#DTree
base_estimator = DecisionTreeClassifier(splitter='random',
                                        max_depth=50,
                                        min_samples_split=8,
                                        min_samples_leaf=8,
                                        )
#NN
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(64, activation='relu')
        self.d3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        y = self.d3(x)
        return y
nn_model = MyModel()
nn_model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                 metrics=['sparse_categorical_accuracy'])

###model_define
tree_model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=500, learning_rate=0.1)
#bp_model = AdaBoostClassifier(base_estimator=nn_model,n_estimators=100)

#train
tree_model.fit(x_train, y_train)
#bp_model.fit(x_train, y_train)

#test
score = tree_model.score(x_test, y_test)
print(F'accuracy:{score}')
y_ = tree_model.predict_proba(x_test)
print("total time:", time() - t, "s")

#plot_metrics
plot_roc('Dtree', 0, n_classes, y_test_b, y_)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()