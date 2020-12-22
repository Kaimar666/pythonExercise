# coding:utf-8
import os

path = r'C:/Users/81275/Desktop/UESTC/datasets/mnist'
filenames = os.listdir(path) #读取path内所有文件名返回列表
i = 0
for filename in filenames:
        sub_path = path + "/" + filename
        f = open(sub_path+'_label.txt', 'w')
        names = os.listdir(sub_path)
        for name in names:
                # splitext方法将文件名和文件类型分开，0是文件名，1是后缀，此处是取文件名的最后一个字符
                label = os.path.splitext(name)[0][-1]
                print(name + ',' + str(label))
                f.write(name + ' ' + str(label) + '\n')