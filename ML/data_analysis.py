"""
1. if cancer 1, density distribution
2. age -> ratio 나이 대비 cancer 비율
3. 스터디 : https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/370333 -> https://www.kaggle.com/code/hengck23/notebooke04a738685
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data 호출
DATA_FOLDER_NAME = "rsna-breast-cancer-detection"
csv_data = pd.read_csv(f"../{DATA_FOLDER_NAME}/train.csv")

# 암인 경우 density의 비중
# NaN A B C D
all_density, density = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
# NaN, 20, 30, 40, 50, 60, 70, 80, 90
all_age, age = [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]

for idx, row in csv_data.iterrows():
    try:
        all_density[ord(row["density"]) - 64] += 1
    except:
        all_density[0] += 1
    # if cancer
    if row["cancer"] == 1:
        try:
            density[ord(row["density"]) - 64] += 1
        except TypeError:
            density[0] += 1

# print(all_density)
# print(density)

ratio = list(map(lambda x: round(x[0] / x[1] * 100, 2), zip(density, all_density)))
# print(ratio)
""" > [1.96, 1.71, 2.44, 2.28, 1.62] density가 cancer에 영향이 있는가? """

for idx, row in csv_data.iterrows():
    try:
        all_age[int((row["age"] // 10) - 1)] += 1
    except:
        all_age[0] += 1
    if row["cancer"] == 1:
        try:
            age[int((row["age"] // 10) - 1)] += 1
        except:
            age[0] += 1

""" age에 따라 cancer의 위험도 상승으로 생각할 수 있다. """
# print(all_age)
# print(age)
age_ratio = list(map(lambda x: round(x[0] / x[1] * 100, 2), zip(age, all_age)))
# print(age_ratio)

import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
import numpy as nd
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.metrics import accuracy_score
dt = deepcopy(csv_data)
# patient_id	image_id	laterality	view	biopsy	invasive	BIRADS	density	difficult_negative_case
feature = dt[dt.columns.difference(["laterality", "view", "cancer", "site_id", "patient_id", "image_id", "biopsy", "invasive", "BIRADS", "density", "difficult_negative_case"])]
y = dt.cancer

xtr, xte, ytr, yte = train_test_split(feature, y, test_size=0.2, random_state=156)
dtrain = xgb.DMatrix(data=xtr, label=ytr)
dtest = xgb.DMatrix(data=xte, label=yte)

params = {'max_depth':3,
         'eta':0.1,
         'objective':'binary:logistic',
         'eval_metric':'logloss',
         'early_stoppings':100}

num_rounds=400

# w_list = [(dtrain, 'train'), (dtest, 'test')]
# xgb_ml = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds,
#                    early_stopping_rounds=100, evals=w_list)
# print(xte)
# pred = xgb_ml.predict(xte)
# print(accuracy_score(yte, pred))
model = XGBClassifier()
model.fit(xtr, ytr)
y_pred = model.predict(xte)
predictions = [round(value) for value in y_pred]

accuray = accuracy_score(yte, predictions)
print(accuray)