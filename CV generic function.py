#!/usr/bin/env python
# coding: utf-8

# In[9]:


# all the libraries we need for this analysis
import imageio 
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pdb
import pandas as pd
from scipy import io
from sklearn import *
import seaborn as sns
import multiprocessing as mp
import scikitplot as skplt
import matplotlib.collections
import statsmodels.api as sm
import pickle 


# In[10]:


def CVgeneric(clf, dat, k, loss_func, train_val_idx, test_dat, features=None,             verbose=True):
    """1. clf: generic classiffer,
     2. dat: training data chunks, 
     3. k: number of folds, 
     4. loss_func: loss function,
     5. train_val_idx: list of available index to select data chunks 
          from data chunks list if only consider data within training 
          set and validation set
     6. test_dat: list of test data in form of [features, labels]
     7. features list to train the model on: features (Default is all 8 features
          excluding x,y coordinates)
     8. verbose: if True, it will print out all loss, accuracy, and 
         test accuracy for all K-fold CV (Default is True)
     
   Output:
     1. K-fold CV loss on the training set(dictionary) in the form of 
          {"CV1":loss} : results
     2. K-fold CV accuracy on the val set(dictionary): accuracy
     3. K-fold CV test accuracy (dictionary): test_accuracy dic
     4. K-fold CV models(list): model_list """
    if features == None:
        features = list(dat[0].columns)[3:]
    num_examples = len(train_val_idx) # for choosing from train_val_idx 
    np.random.seed(1989)
    idx = np.random.permutation(num_examples)
    x_parts = [pd.DataFrame()]*k
    y_parts = [pd.DataFrame()]*k
    chunk_size = num_examples // k #each data chunk's size
    nums_dat = np.sum(list(map(lambda x: x.shape[0], [dat[i] for i in train_val_idx])))
    for i in range(k):
        st_i = i * chunk_size # starting index
        if i == k-1:
            end_i = num_examples 
        else:
            end_i = (i + 1) * chunk_size
        cur_idx = train_val_idx[idx][st_i:end_i]
        cur_x = [dat[i][features] for i in cur_idx] 
        cur_y = [dat[i]["expert label"] for i in cur_idx]
        for x in cur_x:
            x_parts[i] = pd.concat([x_parts[i], x], axis=0)
        for y in cur_y:
            y_parts[i] = pd.concat([y_parts[i], y], axis=0)
    assert np.sum(list(map(lambda x: x.shape[0], (xx for xx in x_parts)))) == nums_dat
    assert np.sum(list(map(lambda x: x.shape[0], (yy for yy in y_parts)))) == nums_dat
    results = {}
    accuracy = {}
    test_acc_dic = {}
    model_list = []
    workers = mp.Pool()
    for i in range(0, k):
        X_val, Y_val = x_parts[i].values, y_parts[i].values.reshape(-1,)
        X_train = pd.DataFrame()
        Y_train = pd.DataFrame()
        for cur in x_parts[:i]+x_parts[i+1:]:
            X_train = pd.concat([X_train, cur], axis=0)
        assert X_train.shape[0] + X_val.shape[0] == nums_dat
        for cur in y_parts[:i]+ y_parts[i+1:]:
            Y_train = pd.concat([Y_train, cur], axis=0)
        assert Y_train.shape[0] + Y_val.shape[0] == nums_dat
        w = workers.apply_async(clf.fit, args=(X_train.values,                                            Y_train.values.reshape(-1,)))
        model = w.get()
        model_list.append(model)
        if loss_func is metrics.log_loss:
            pred = model.predict_proba(X_val)
        else:
            pred = model.predict(X_val)
        loss = loss_func(Y_val, pred)
        results["CV%d" %(i+1)] =  loss
        acc = metrics.accuracy_score(Y_val,model.predict(X_val))
        accuracy["CV%d" %(i+1)] =  acc
        test_acc = metrics.accuracy_score(test_dat[1], model.predict(test_dat[0][features]))
        if verbose:
            print("CV", i+1, "mean loss: ", np.mean(loss))
            print("Accuracy for CV", i+1, ": %",acc*100)
            print("Test Accuracy for CV", i+1, ": %",test_acc*100)
            print(" ")
        test_acc_dic["CV%d" %(i+1)] =  test_acc
    workers.close()
    return results, accuracy, test_acc_dic, model_list


# In[ ]:




