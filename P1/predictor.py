#! /usr/bin/python

import sys
import numpy as np
from sys import argv
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

sample = [sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]]
sample = np.array(sample).reshape(1,-1)

b_model = joblib.load('best_model.pkl')

iris = datasets.load_iris()

b_model.fit(iris.data, iris.target)
predict = b_model.predict(sample)
avg = np.mean(predict)
print (avg)
