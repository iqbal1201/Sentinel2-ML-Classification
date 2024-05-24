
#Importing all modules required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.utils import resample
import time, os, shutil
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


## CHANGE REQUIRED HERE!!!
## Do this to import dataframe containing training point depending on how many datasets you have

area1 = pd.read_csv('Retraining_banten_index.csv') 
area2 = pd.read_csv('Retraining_sumbar_index.csv')
area3 = pd.read_csv('Retraining_kalteng_index.csv')
area4 = pd.read_csv('Retraining_medan_index.csv')

## CHANGE REQUIRED HERE!!!
# Appending all datasets into one dataframe. Do this if you get new training data and in separated files
new_training = area1.append(area2).append(area3).append(area4)




## DO NOT CHANGE


# Choosing the feature (correct columns) for ML training
new_training = new_training[["Label", "MBI", "MNDWI", "NDVI", "SAVI"]]


# Encoding

# Encoding old character label into numerical value. Bareland and BuiltUp is merged into one. 
# This is optional but depending on the business needs
new_training.Label[new_training.Label=="Bareland"] = 1
new_training.Label[new_training.Label=="BuiltUp"] = 1
new_training.Label[new_training.Label=="Vegetation"] = 2
new_training.Label[new_training.Label=="WaterBody"] = 3
new_training.Label[new_training.Label=="Cloud"] = 4


# Drop Null Value
new_training.dropna(inplace=True)

# Dataset Splitting

X = new_training[['MBI', 'MNDWI', 'NDVI', 'SAVI']]
y = new_training.Label.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Model Training

num_pipe = Pipeline([
    ('scaler', MinMaxScaler())
])

preprocessor = ColumnTransformer([
    ('numeric', num_pipe, X_train.columns)
])


pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', LogisticRegression(multi_class='multinomial', solver='sag', n_jobs=-1, random_state=42))
])

logreg_param = {
    'algo__fit_intercept' : [True],
    'algo__C' : np.array([1e2])
}

model_logreg = LogisticRegression(multi_class='multinomial', solver='sag', n_jobs=-1, random_state=42)


X_train.info()


model = GridSearchCV(pipeline, logreg_param, cv=5, n_jobs=-1, verbose=1)
model_logreg.fit(X_train, y_train)



#Make Prediction

y_pred = model_logreg.predict(X_test)
print(classification_report(y_test,y_pred))






## CHANGE REQUIRED HERE!!!
## Save Dataset & Model
# saving all appended dataframe to new dataframe 
new_training.to_csv('new_training19052022.csv', index=False)

with open('model_logreg_05192022.sav', 'wb') as f:
    pickle.dump(model_logreg, f)

os.walk('C:\Users\mjanuadi\Documents\Development ML\Sample Wilmar')

path

