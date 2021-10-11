from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils import _deprecate_positional_args, check_matplotlib_support
from scipy import stats

# acc_hc = acc[0:44]
# acc_add = acc[44:96]
# acc_adhd = acc[96:144]           #数据由id划分为三类
# acc = np.array(acc)
# acc_hc = np.array(acc_hc)
# acc_add = np.array(acc_add)
# acc_adhd = np.array(acc_adhd)

# print('acc_hc: ', acc_hc.mean())
# print('acc_add', acc_add.mean())
# print('acc_adhd', acc_adhd.mean())
# print('acc_mean: ', acc.mean())
# print('acc_std: ', np.std(acc))

# model acc 由此前 txt 记录得到，可以将测试数据结果输入到[]矩阵中
acc_EEGNet = []
acc_ShallowCNN = []
acc_DeepCNN = []

# cnnlstm model LOOS验证结果的准确率得到例子 如下
result1 = []
for i in range(144):
    print(i + 1)
    result1_ = pd.read_csv(r'C:\Users\siat-sj1\Desktop\SCIresult\model_result\leaveOneOutResult\CNNLSTM3\LOOS-result-'
                           + str(i + 1) + '.txt', delimiter='\t')
    result1_ = np.array(result1_)
    # print(result1_.shape)
    pred = result1_[:, 4]
    real = result1_[:, 5]
    acc_ = np.mean(pred == real)
    result1.append(acc_)
acc = [result1, acc_shallowCNN, acc_DeepCNN, acc_EEGNET]
acc = np.array(acc)

F, P = stats.f_oneway(result1, acc_shallowCNN, acc_DeepCNN, acc_EEGNET)
print('F', F)
print('p', P)

for i in range(144):
    result1[i] = float(format(result1[i], '.4f'))
print(result1)
# print(np.mean(result1))
# plt.bar(np.arange(144), acc)
# plt.ylim(0.0, 1.0)
# plt.show()

# output_dir = r'E:\wangcheng\Experiment-01-out'


#========================================================== 单个模型的混淆矩阵，敏感性，特异性，auc指标等计算如下================================
def Confusion_matrix_plot(y_true, y_pred):
    class_names = ['HC', 'ADD', 'ADHD']
    cm = confusion_matrix(y_true, y_pred)
    cm_disp = Cmdisplay(cm, display_labels=class_names)
    cm_disp.plot(cmap='Blues')
    # confusion_matrix_save_path = os.path.join(output_dir, 'confusion_matrix.png')
    # plt.savefig(confusion_matrix_save_path, dpi=300)  # dpi分辨率
    print('plot done...........')
    plt.show()

    return cm


class Cmdisplay:
    @_deprecate_positional_args
    def __init__(self, confusion_matrix, *, display_labels=None):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    @_deprecate_positional_args
    def plot(self, *, include_values=True, cmap='viridis',
             xticks_rotation='horizontal', values_format=None,
             ax=None, colorbar=True):
        check_matplotlib_support("ConfusionMatrixDisplay.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap)
        self.text_ = None
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)

            # print text with appropriate color depending on background
            thresh = (cm_norm.max() + cm_norm.min()) / 2.0

            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm_norm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], '.4f')
                    text_cm_norm = format(cm_norm[i, j], '.4f')
                    if cm.dtype.kind != 'f':
                        text_d = format(cm[i, j], 'd')
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)
                    text_cm_norm = format(cm_norm[i, j], values_format)

                text_cm = np.array(text_cm)
                self.text_[i, j] = ax.text(
                    j, i, '(' + str(text_cm) + ')', fontsize=13,
                    ha="center", va="top",
                    color=color)
                self.text_[i, j] = ax.text(
                    j, i, text_cm_norm, fontsize=13,
                    ha="center", va="bottom",
                    color=color)

        if self.display_labels is None:
            display_labels = np.arange(n_classes)
        else:
            display_labels = self.display_labels
        if colorbar:
            fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=display_labels,
               yticklabels=display_labels, )

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.set_xlabel("Prediction", fontsize=15)
        ax.set_ylabel("GroundTruth", fontsize=15)

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self

# data = []
# for i in range(5):
#     data_1 = pd.read_csv(
#         r'C:\Users\siat-sj1\Desktop\SCIresult\model_result\test-430-shallowcnn\k-fold-result-' + str(i) + '.txt',
#         delimiter='\t')
#     data.append(data_1)
#
# data = np.array(data).reshape(5 * 6000, 6)
# pred_y = data[:, 4]
# true_y = data[:, 5]
# print('pred_y: ', pred_y)
# print('true_y: ', true_y)

# cm = Confusion_matrix_plot(true_y, pred_y)

# cm_norm = cm / cm.sum(axis=1, keepdims=True)
# print('cm.shape: ', cm.shape)
# print('cm: ', cm)
# FP = cm.sum(axis=0) - np.diag(cm)
# FN = cm.sum(axis=1) - np.diag(cm)
# TP = np.diag(cm)
# TN = cm.sum() - (FP + FN + TP)
# print(TP)
# print(TN)
# print(FP)
# print(FN)
# FP = FP.astype(float)
# FN = FN.astype(float)
# TP = TP.astype(float)
# TN = TN.astype(float)
#
# acc = accuracy_score(pred_y, true_y)
# sens = TP / (TP + FN)  # sensitivity / recall
# spec = TN / (FP + TN)  # specificity
# ppv = TP / (TP + FP)  # positive predictive value 阳性预测率 / precision
# npv = TN / (TN + FN)  # negative predictive value阴性预测率
# mcc = (TP * TN - FP * FN) / np.sqrt(
#     ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))  # Matthew's correlation coefficient 相关系数
# F1 = 2 * (sens * ppv) / (sens + ppv)
#
# score_y = data[:, 1:4]
# print(score_y)
# fpr = dict()
# tpr = dict()
# auc_ = dict()
# for i in range(3):
#     fpr[i], tpr[i], _ = roc_curve(true_y, score_y[:, i], pos_label=i)
#     auc_[i] = auc(fpr[i], tpr[i])  # calculate auc
#
# print('acc is ', acc)
# print('sens is ', sens.mean())
# print('spec is ', spec.mean())
# print('ppv is ', ppv.mean())
# print('npv is ', npv.mean())
# print('mcc is ', mcc.mean())
# print('auc is ', auc_[0], auc_[1], auc_[2])
# print('auc.mean is ', ((auc_[0] + auc_[1] + auc_[2]) / 3))
# print('f1score is ', F1.mean())



# ---------------------------------------------ROC curve----------------------------------------------------------------#
# lw = 2  # 定义线条宽度
# plt.figure(figsize=(8, 5))
# plt.plot(fpr[0], tpr[0], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.4f)' % auc_[0])  ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic curve of PFO classification')
# plt.legend(loc="lower right")
# # ROC_curve_save_path = os.path.join(input_dir, 'ROC_curve.png')
# # plt.savefig(ROC_curve_save_path, dpi=300)#dpi分辨率
# plt.show()
