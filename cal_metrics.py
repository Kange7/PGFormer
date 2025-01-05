# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to compute metrics on any dataset
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Based on the work of Puzuo Wang
#

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
from utils.ply import read_ply

def Cal_from_confusions(confusions):
    """
    Computes IoU from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) IoU score
    """

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    precision = TP / (TP_plus_FP + 1e-6)
    recall = TP / (TP_plus_FN + 1e-6)
    f1 = (2.0*precision*recall)/(precision+recall + 1e-6)
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU

    precision = np.append(precision, np.mean(precision))
    recall = np.append(recall, np.mean(recall))
    f1 = np.append(f1, np.mean(f1))
    iou = np.append(IoU, np.mean(IoU))

    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    print('Precision: ', 100.*precision)
    print('Recall: ', 100.*recall)
    print('F1: ', 100.*f1)
    print('IOU: ', 100.*iou, '\n')

    return 100.*f1[-1]


if __name__ == '__main__':


    # H3D
    print('Dataset: H3D')
    logs =[

        'Log_2024-12-17_14-05-42(H3D)',
    
    ]
    
    for log in logs:
        print(log)
        ply_file = '../test/' + log + '/predictions/Mar18_test.ply'
        # ply_file = '../test/Log_2024-11-20_06-06-21(K-ISPRS)/predictions/Vaihingen3D_EVAL_WITH_REF.ply'
        data = read_ply(ply_file)
        gp = np.vstack((data['gt'], data['preds'])).T
        oa = accuracy_score(gp[:, 0], gp[:, 1])
        print('Overall Accuracy: {:.2f}%'.format(100. * oa))
        cm = confusion_matrix(gp[:, 0], gp[:, 1])
        Cal_from_confusions(cm)

