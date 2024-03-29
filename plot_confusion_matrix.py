import itertools
import matplotlib.pyplot as plt  # 绘图库
import numpy as np
import os

def plot_confusion_matrix(cm, labels_name, title, acc, save_path):
    cm = cm / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    print(cm)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.imshow(cm, interpolation='nearest',cmap=plt.get_cmap('YlOrBr'))  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_class = np.array(range(len(labels_name)))  # 获取标签的间隔数
    plt.xticks(num_class, labels_name)  # 将标签印在x轴坐标上
    plt.yticks(num_class, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('Target')
    plt.xlabel('Prediction')
    # plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('YlOrBr'))
    plt.tight_layout()
    # 0:surprise, 1:fear, 2:disgust,
    # 3: happiness, 4: sad, 5: angry, 6: neu
    plt.savefig(os.path.join(save_path, "acc" + str(acc) + ".png"), format='png')
    plt.show()
    plt.close()

