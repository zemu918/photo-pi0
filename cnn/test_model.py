# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,roc_curve,auc,precision_recall_curve,average_precision_score,classification_report,recall_score
import matplotlib.pyplot as plt
import pickle

from config import MODEL_DIR, DATA_DIR, OUTPUT_DIR

from plot_cm import plot_cm

task_name = sys.argv[1]

##np.set_printoptions(threshold = np.inf)

def get_dimension(lst):
    if isinstance(lst, list):
        return[len(lst)] + get_dimension(lst[0])
    else:
        return[]

def my_softmax(x):
    a = np.max(x)
    exp_a = np.exp(x - a)
    sum_exp_a = np.sum(exp_a)
    return exp_a/sum_exp_a

def prob2label(probas,threshold):
    temp=[]
    for i in probas:
        if i >= threshold:
            temp.append(1)
        else:
            temp.append(0)
    return temp

def evaluate_cnn(ttask_nameask_name):
    print(f"Loading data for task {task_name}...")
    with open(f"{DATA_DIR}/processed/cnn/{task_name}_X_test.pkl", "rb") as fin:
        X_test = pickle.load(fin)
    with open(f"{DATA_DIR}/processed/cnn/{task_name}_Y_test.pkl", "rb") as fin:            
        Y_test = pickle.load(fin)
    print(f"Loading model...")

    cnn = keras.models.load_model(f"{MODEL_DIR}/{task_name}_accuracy_cnn.h5")

##    print("Y_test type: ",type(Y_test))
##    print("Y_test shape: ",Y_test.shape)
##    print("Y_test: ",Y_test)
##    print("X_test type: ",type(X_test))
##    print("X_test shape: ",X_test[0].shape)
##    print("X_test: ",X_test[0])
##
    Y_pred = cnn.predict(X_test)
    print("Y_pred type: ",type(Y_pred))
    print("Y_pred shape: ",Y_pred.shape)
    print("Y_pred: ",Y_pred)

    origin1 = Y_pred[...,0]
    origin2 = Y_pred[...,1]

##    class_all = tf.nn.softmax(Y_pred)
##    class_all = class_all.numpy()
##    print("class1 type: ",type(class_all))
##    print("class1 shape: ",class_all.shape)
##    print("class1: ",class_all)
##    class1 = class_all[...,0]
##    class2 = class_all[...,1]
##
##    print("class1 type: ",type(class1))
##    print("class1 shape: ",class1.shape)
##    print("class1: ",class1)
##    print("class2 type: ",type(class2))
##    print("class2 shape: ",class2.shape)
##    print("class2: ",class2)   

##    arr=Y_pred
##    class_prob = np.apply_along_axis(my_softmax,0,arr)
##    #0 means column

    #### Draw compare diag  origin1(no softmax)/class1(softmax)
##    array1 = []
##    array2 = []
##    for index, value in enumerate(Y_test):
##        if value == 0:
##            array1.append(origin1[index])
##        elif value == 1:
##            array2.append(origin1[index])
##        else:
##            print("There may some error")
##
##    array1 = np.array(array1)
##    array2 = np.array(array2)
##    if len(origin1) == (len(array1) + len(array2)):
##        print("everything is OK!")
##
##    bins =np.linspace(0,1,200)
##    plt.hist(array1,bins,alpha=0.5, label="signal")
##    plt.hist(array2,bins,alpha=0.5, label="back")
##    plt.xlabel('probability')
##    plt.ylabel('event number')
##    plt.legend(loc="upper left")
##    plt.show()
    
    Accuracy = []
    Precison = []
    thre_list = []
    Misidf = []
    thresholds = np.arange(0,1,0.00001)
    for thr in thresholds:
        class_loop = prob2label(origin1,thr)
        class_loop = np.array(class_loop)
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        accuracy = 0.0
        precision = 0.0
        if thr == 0:
            continue
        for index,cla in enumerate(class_loop):
            if cla == 1 and Y_test[index] == 1:
                FP += 1
            elif cla == 1 and Y_test[index] == 0:
                TP += 1
            elif cla == 0 and Y_test[index] == 1:
                TN += 1
            elif cla == 0 and Y_test[index] == 0:
                FN += 1
        if (TP+FP) == 0: continue 
        if (TN+FN) == 0: continue 
        accuracy = (TP+TN)/(TP+FP+FN+TN)
        precision = TP/(TP+FP)
        misidf = FN/(TN+FN)

        Accuracy.append(accuracy)
        Precison.append(precision)
        thre_list.append(thr)
        Misidf.append(misidf)

    np.savetxt(f'../precision/{task_name}_evaluate.txt',np.column_stack((Accuracy,Precison,Misidf,thre_list)),fmt='%.6f %.6f %.6f %.3f',delimiter="   ")

##    ### confusion matrix
##    Y_pred = np.argmax(cnn.predict(X_test, batch_size=10), axis=1)
##    
##    cm = confusion_matrix(Y_test, Y_pred).astype(float)
##    cm /= np.sum(cm, axis=1, keepdims=True)
##    
##    labels = [r"$2\pi^0$", r"$nogam$"]
##    
##    os.makedirs(f"{OUTPUT_DIR}/cnn_results", exist_ok=True)
##    plot_cm(
##        cm,
##        labels=labels,
##        save_path=f"{OUTPUT_DIR}/cnn_results/{task_name}_CNN_ConfusionMatrix.pdf"
##    )
##    test_accuracy = np.mean(Y_pred == Y_test)
##    print(f"Mean test accuracy for {task_name}: {test_accuracy:.5f}")

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
##    plt.show()

    #### P-R curve
    #scores = cnn.evaluate(X_test,Y_test)
    #print(f"score dim= {get_dimension(scores)}")
    #print(f"test dim = {Y_test.shape},{X_test.shape}")
    #Y_pred_pr = cnn.predict(X_test)[:, 1]

##    precision,recall,thresholds = precision_recall_curve(Y_test, class1)
##    for idx,x in enumerate(thresholds):
##        print(x,"   ", precision[idx])
    

##    ap = average_precision_score(Y_test, Y_pred_pr)
##    plt.plot(recall, precision, label='PR Curve (AUC = %0.2f)'.format(ap))
##    plt.xlim([0.0, 1.0])
##    plt.ylim([0.0, 1.05])
##    plt.xlabel('Recall')
##    plt.ylabel('Precision')
##    plt.title('Precision-Recall Curve')
##    plt.legend(loc="lower left")
##    plt.show()

##    return test_accuracy, cm

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
