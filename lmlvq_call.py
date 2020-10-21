#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:49:35 2020

@author: avinash
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from lmlvq_distance import LMLVQ

prototype_per_class = 1

input_data = load_iris().data
data_label = load_iris().target
epochs = 10
learning_rate = 0.001
margin = 3
C = 10
kappa = 2

clf = LMLVQ(prototype_per_class)

X_train, X_test, y_train, y_test = train_test_split(input_data,
                                                    data_label,
                                                    test_size=0.3,
                                                    random_state=42)

clf.fit(X_train, y_train, learning_rate, epochs, margin, C, kappa)

y_predict = clf.predict(X_test, X_train)
print("Predicted result: ", y_predict)
print("Actual result: ", y_test)

acc = np.sum(y_predict == y_test)
acc = acc / len(y_test) * 100
print("Accuracy is: ",acc)
