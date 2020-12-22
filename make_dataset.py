import os
import numpy as np
from PIL import Image

train_path = r"C:/Users/81275/Desktop/UESTC/datasets/mnist/train/"
train_txt = r"C:/Users/81275/Desktop/UESTC/datasets/mnist/train_label.txt"
x_train_savepath = r"C:/Users/81275/Desktop/UESTC/datasets/mnist/x_train.npy"
y_train_savepath = r"C:/Users/81275/Desktop/UESTC/datasets/mnist/y_train.npy"

test_path = r"C:/Users/81275/Desktop/UESTC/datasets/mnist/test/"
test_txt = r"C:/Users/81275/Desktop/UESTC/datasets/mnist/test_label.txt"
x_test_savepath = r"C:/Users/81275/Desktop/UESTC/datasets/mnist/x_test.npy"
y_test_savepath = r"C:/Users/81275/Desktop/UESTC/datasets/mnist/y_test.npy"

def generateds(path, txt):
    f = open(txt, 'r')
    contents = f.readlines()
    f.close()
    x, y = [], []
    for content in contents:
        value = content.split() #以空格分开,value[0]是图片名，value[1]是标签
        img_path = path + value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L')) #L是指灰度图
        img = img / 255. #做归一化处理
        x.append(img)
        y.append(value[1])
        print('loading' + content)

    x = np.array(x)
    y = np.array(y).astype(np.int64)
    return x, y

if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(x_test_savepath) and os.path.exists(y_test_savepath):
    print('---------------loadDatasets---------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    #x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
    #x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))

else:
    print('--------------GenerateDatasets------------')
    x_train, y_train = generateds(train_path, train_txt)
    x_test, y_test = generateds(test_path, test_txt)

    print('---------------SaveDatasets---------------')
    x_train_save = np.reshape(x_train,(len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(x_test_savepath, x_test_save)
    np.save(y_train_savepath, y_train)
    np.save(y_test_savepath, y_test)