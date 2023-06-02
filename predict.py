from pickle import load
# packages
import os
import gc
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc, f1_score, accuracy_score
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier



with open('./best_dt.pkl', 'rb') as f:
	best_dt = load(f)

with open('./oe_equip.pkl', 'rb') as f:
	oe_equip = load(f)

with open('./oe_mold.pkl', 'rb') as f:
	oe_mold = load(f)

with open('./scaler.pkl', 'rb') as f:
	scaler = load(f)


in_radius = float(input("IN_RADIUS 입력 (절댓값으로): "))
out_radius = float(input("OUT_RADIUS 입력 (절댓값으로): "))
power = float(input("POWER 입력 : "))
equip_id = input("EQUIP_ID 입력 (ex.EQUIP3) : ")
mold_pos = int(input("MOLD_POS 입력 : "))


data = pd.DataFrame([[equip_id,mold_pos,in_radius,out_radius,power]],columns = ['EQUIP_ID','MOLD_POS','IN_RADIUS','OUT_RADIUS','POWER'])

data_tmp = data[['IN_RADIUS','OUT_RADIUS','POWER']]

encoded_equip = oe_equip.transform(data[['EQUIP_ID']])
encoded_mold = oe_mold.transform(data[['MOLD_POS']])

data_encoded_equip = pd.DataFrame(encoded_equip, columns=['EQUIP_ID_' + col for col in oe_equip.categories_[0]])
data_encoded_mold = pd.DataFrame(encoded_mold, columns=['MOLD_POS_' + str(col) for col in oe_mold.categories_[0]])

data_encoded = pd.concat([data_tmp, data_encoded_equip, data_encoded_mold],axis=1)

print(data_encoded)
new_test = scaler.transform(data_encoded)
new_pred = best_dt.predict(new_test)

print("Prediction")
if new_pred == 0:
	print("Normal")
else:
	print("Anomaly")


