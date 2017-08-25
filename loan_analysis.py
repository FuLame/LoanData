import pandas as pd
import matplotlib.pyplot as plt

from data_transform import *
import my_models as mm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from time import time
import warnings

#Settings
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore")

#Load data to dataframe df
#After the first run of the script the data is transformed and saved.
#Future runs restore already transformed data, which saves time and memory.
try:
    df = pd.read_csv("loan_short.csv", low_memory=False)
except:
    df = pd.read_csv("loan.csv", low_memory=False)
    #Pre-process data
    df = transform_data(df)
    df.to_csv("loan_short.csv")

#Separate data into features X and labels Y
Y = df.label.copy()
df.drop(['label'], axis=1, inplace=True)
X = df.copy()

#Process data
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.3,random_state=42)
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

#Train models
#Logistic Regression Classifier
t0 = time()
fpr_lr, tpr_lr, score_lr = mm.LogReg(train_x, train_y, test_x, test_y)
print("lr",round(time()-t0),score_lr)

# Random Forest
t0 = time()
fpr_rf, tpr_rf, score_rf = mm.RandFor(train_x, train_y, test_x, test_y)
print("rf", round(time() - t0), score_rf)

# Gradient Boosting
t0 = time()
fpr_gb, tpr_gb, score_gb = mm.GradBoost(train_x, train_y, test_x, test_y)
print("gb", round(time() - t0), score_gb)

# Gradient Boosting with Linear Regression
t0 = time()
fpr_gb_lr, tpr_gb_lr, score_gb_lr = mm.GradBoost_LR(train_x, train_y, test_x, test_y)
print("gb_lr", round(time() - t0), score_gb_lr)

#Neural Network
t0 = time()
fpr_nn, tpr_nn, score_nn = mm.FFNN(
    train_x, train_y, test_x, test_y,cv=False)
print("nn",round(time()-t0),score_nn)

#Plotting the ROC curves with AUC score
plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_lr, tpr_lr, label="LR"+" auc: "+str(score_lr))
plt.plot(fpr_rf, tpr_rf, label="RF"+" auc: "+str(score_rf))
plt.plot(fpr_gb, tpr_gb, label="GB"+" auc: "+str(score_gb))
plt.plot(fpr_gb_lr, tpr_gb_lr, label="GB+LR"+" auc: "+str(score_gb_lr))
plt.plot(fpr_nn, tpr_nn, label="NN"+" auc: "+str(score_nn))

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc='best')
plt.title("ROC curves")
plt.show()


