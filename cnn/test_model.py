# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import json
import numpy as np
import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,roc_curve,auc,precision_recall_curve,average_precision_score
import matplotlib.pyplot as plt
import pickle

from config import MODEL_DIR, DATA_DIR, OUTPUT_DIR

from plot_cm import plot_cm

task_name = sys.argv[1]

def get_dimension(lst):
    if isinstance(lst, list):
        return[len(lst)] + get_dimension(lst[0])
    else:
        return[]

def evaluate_cnn(task_name):
    print(f"Loading data for task {task_name}...")
    with open(f"{DATA_DIR}/processed/cnn/{task_name}_X_train.pkl", "rb") as fin:
        X_test = pickle.load(fin)
    with open(f"{DATA_DIR}/processed/cnn/{task_name}_Y_train.pkl", "rb") as fin:        
        Y_test = pickle.load(fin)
    print(f"Loading model...")
    cnn = keras.models.load_model(f"{MODEL_DIR}/nogam_cnn_huge.h5")
    
    ### confusion matrix
    Y_pred = np.argmax(cnn.predict(X_test, batch_size=10), axis=1)
    
    cm = confusion_matrix(Y_test, Y_pred).astype(float)
    cm /= np.sum(cm, axis=1, keepdims=True)
    
    labels = [r"$2\pi^0$", r"$nogam$"]
    
    os.makedirs(f"{OUTPUT_DIR}/cnn_results", exist_ok=True)
    plot_cm(
        cm,
        labels=labels,
        save_path=f"{OUTPUT_DIR}/cnn_results/{task_name}_CNN_ConfusionMatrix.pdf"
    )
    test_accuracy = np.mean(Y_pred == Y_test)
    print(f"Mean test accuracy for {task_name}: {test_accuracy:.5f}")

##    ### roc curve 
##    Y_pred_keras = cnn.predict(X_test)[:, 1]
##    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test,Y_pred_keras)
##    auc_keras = auc(fpr_keras, tpr_keras)
##    plt.figure()
##    plt.plot([0, 1], [0, 1], 'k--')
##    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
##    plt.xlabel('False positive rate')
##    plt.ylabel('True positive rate')
##    plt.title('ROC curve')
##    plt.legend()
    plt.show()

    #### P-R curve
    #scores = cnn.evaluate(X_test,Y_test)
    #print(f"score dim= {get_dimension(scores)}")
    #print(f"test dim = {Y_test.shape},{X_test.shape}")
##    Y_pred_pr = cnn.predict(X_test)[:, 1]
##    precision,recall,thresholds = precision_recall_curve(Y_test, Y_pred_pr)
##    ap = average_precision_score(Y_test, Y_pred_pr)
##    plt.plot(recall, precision, label='PR Curve (AUC = %0.2f)'.format(ap))
##    plt.xlim([0.0, 1.0])
##    plt.ylim([0.0, 1.05])
##    plt.xlabel('Recall')
##    plt.ylabel('Precision')
##    plt.title('Precision-Recall Curve')
##    plt.legend(loc="lower left")
##    plt.show()

    return test_accuracy, cm

evaluate_cnn(task_name)

"""
    ### roc curve 
    Y_pred_keras = cnn.predict(X_test)[:, 1]
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test,Y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()
"""    
