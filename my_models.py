from sklearn.ensemble import RandomForestClassifier,\
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV,\
    train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc

from tf_nn import evaluate
import numpy as np
import pandas as pd

#Feed Forward Neural Network
def FFNN(train_x, train_y, test_x, test_y,cv=False):
    '''
    Creates and fits the Neural Network
    :param train_x: train_x
    :param train_y: train_y
    :param test_x: test_x
    :param test_y: test_y
    :param cv: bool: If cross-validation should be performed
    :return: fpr, tpr, auc_score
    '''
    if cv:
        scores = []
        best_params = pd.DataFrame(columns=['beta','n','score'])
        for beta in [0.01, 0.05, 0.1]:
            for n in [50,100,200]:
                #Get the size of the training set to split
                size = int(len(train_x)/3)
                X = np.array([ train_x[:size],train_x[size:2*size],train_x[2*size:] ])
                Y = np.array([ train_y[:size],train_y[size:2*size],train_y[2*size:] ])
                for i in range(3):
                    #Split the data into cross-val set and train set:
                    #One third into the cross-val set
                    cv_x, cv_y = X[i], Y[i]
                    #Remaining into the training set
                    x_train_cv = np.vstack( (X[(i+1)%3], X[(i+2)%3]) )
                    y_train_cv = np.hstack( (Y[(i+1)%3], Y[(i+2)%3]) )
                    #Evaluate the model
                    predictions = evaluate(x_train_cv, y_train_cv,cv_x,cv_y,beta,n)
                    fpr, tpr, _ = roc_curve(cv_y, predictions, pos_label=1.0)
                    score = auc(fpr, tpr, reorder=True)
                    #Add the score to the list to average
                    scores.append(score)
                #For each set of parameters create an entry
                best_params = best_params.append(pd.DataFrame({
                    'beta':[beta,],'n':[n,],
                    'score':[np.array(scores).mean(),]}),
                    ignore_index=True)
        #Obtain a row with the highest cross-val score
        best_row = best_params.loc[best_params['score'].idxmax()]
        beta, n = best_row.beta, int(best_row.n)

        #Fit the NN with the best params beta and n
        predictions = evaluate(train_x, train_y, test_x, test_y, beta, n)
        fpr, tpr, _ = roc_curve(test_y, predictions, pos_label=1.0)
        score = auc(fpr, tpr, reorder=True)

    else:
        #Fit the NN with optimal beta and n parameters
        predictions = evaluate(train_x, train_y, test_x, test_y, 0.05, 100)
        fpr, tpr, _ = roc_curve(test_y, predictions, pos_label=1.0)
        score = auc(fpr, tpr, reorder=True)
    return fpr, tpr, score

#Logistic Regression Classifier
def LogReg(train_x, train_y, test_x, test_y,parameters=None):
    '''
    Creates and fits the Logistic Regression Classifier
    by GridSearchCV
    :param train_x: train_x
    :param train_y: train_y
    :param test_x: test_x
    :param test_y: test_y
    :param parameters: *Dict for GridSearchCV
    :return: fpr, tpr, auc_score
    '''
    if parameters == 'default':
        parameters = {'C':[0.1,0.3,1.0,3.0,10.0,30.0,100.0,300.0]}
        lr = LogisticRegression(n_jobs=-1)
        clf_lr = GridSearchCV(lr, parameters, scoring='roc_auc')
    elif type(parameters) == dict:
        lr = LogisticRegression(n_jobs=-1)
        clf_lr = GridSearchCV(lr,parameters,scoring='roc_auc')
    else:
        clf_lr = LogisticRegression(C=100.0, n_jobs=-1)

    clf_lr.fit(train_x,train_y)
    predictions = clf_lr.predict_proba(test_x)[:, 1]
    fpr, tpr, _ = roc_curve(test_y, predictions, pos_label=1.0)
    score = round(roc_auc_score(test_y, predictions),4)

    return fpr, tpr, score


#Gradient Boosting Classifier
def GradBoost(train_x, train_y, test_x, test_y, parameters=None):
    '''
    Creates and fits the Gradient Boosting Classifier
    by GridSearchCV
    :param train_x: train_x
    :param train_y: train_y
    :param test_x: test_x
    :param test_y: test_y
    :param parameters: *Dict for GridSearchCV
    :return: fpr, tpr, auc_score
    '''
    if parameters == 'default':
        parameters = {"max_depth":[3,4],
                      "n_estimators":range(100,600,50)}
        gb = GradientBoostingClassifier()
        clf_gb = GridSearchCV(gb, parameters, scoring='roc_auc')
    elif type(parameters) == dict:
        gb = GradientBoostingClassifier()
        clf_gb = GridSearchCV(gb,parameters,scoring='roc_auc')
    else:
        clf_gb = GradientBoostingClassifier(max_depth=4, n_estimators = 500)

    clf_gb.fit(train_x,train_y)
    predictions = clf_gb.predict_proba(test_x)[:,1]
    fpr, tpr, _ = roc_curve(test_y, predictions, pos_label=1.0)
    score = round(roc_auc_score(test_y, predictions),4)

    return fpr, tpr, score


#Random Forest Classifier
def RandFor(train_x, train_y, test_x, test_y,parameters=None):
    '''
    Creates and fits the Random Forest Classifier
    by GridSearchCV
    :param train_x: train_x
    :param train_y: train_y
    :param test_x: test_x
    :param test_y: test_y
    :param parameters: *Dict for GridSearchCV
    :return: fpr, tpr, auc_score
    '''
    if parameters == 'default':
        parameters = {"n_estimators":range(100,800,50)}
        rf = RandomForestClassifier(n_jobs=-1)
        clf_rf = GridSearchCV(rf, parameters, scoring='roc_auc')
    elif type(parameters) == dict:
        rf = RandomForestClassifier(n_jobs=-1)
        clf_rf = GridSearchCV(rf,parameters,scoring='roc_auc')
    else:
        #Optimal parameters obtained previously from GridSearchCV
        clf_rf = RandomForestClassifier(n_estimators=600, n_jobs=-1)

    clf_rf.fit(train_x,train_y)
    predictions = clf_rf.predict_proba(test_x)[:,1]
    fpr, tpr, _ = roc_curve(test_y, predictions, pos_label=1.0)
    score = round(roc_auc_score(test_y, predictions),4)
    return fpr, tpr, score


#Stochastic Gradient Descent Classifier
def SGDC(train_x, train_y, test_x, test_y,parameters=None):
    '''
    Creates and fits the SGDClassifier
    :param train_x: train_x
    :param train_y: train_y
    :param test_x: test_x
    :param test_y: test_y
    :return: fpr, tpr, auc_score
    '''
    clf_sgd = SGDClassifier(n_jobs=-1)
    clf_sgd.fit(train_x,train_y)
    predictions = clf_sgd.decision_function(test_x)
    fpr, tpr, _ = roc_curve(test_y, predictions, pos_label=1.0)
    score = round(roc_auc_score(test_y, predictions),4)
    return fpr, tpr, score


#Gradient Boosting Classifier with Linear Regression
def GradBoost_LR(train_x, train_y, test_x, test_y, parameters=None):
    '''
    Creates and fits the Gradient Boosting Classifier
    by GridSearchCV
    :param train_x: train_x
    :param train_y: train_y
    :param test_x: test_x
    :param test_y: test_y
    :param parameters: *Dict for GridSearchCV
    :return: fpr, tpr, auc_score
    '''
    #Logistic Regression should be fitted on a different set to
    #prevent overfitting
    train_x, train_x_lr, train_y, train_y_lr = train_test_split(
        train_x, train_y, test_size=0.3, random_state=42)

    if parameters == 'default':
        parameters = {"max_depth": [3, 4],
                  "n_estimators": range(100, 600, 50)}
        gb = GradientBoostingClassifier()
        clf_gb = GridSearchCV(gb, parameters, scoring='roc_auc')
    elif type(parameters) == dict:
        gb = GradientBoostingClassifier()
        clf_gb = GridSearchCV(gb,parameters,scoring='roc_auc')
    else:
        #Optimal parameters obtained previously from GridSearchCV
        clf_gb = GradientBoostingClassifier(n_estimators=500)

    #Define one hot encoder to transform the output of GB to fit
    #the input of LR
    enc_gb = OneHotEncoder()
    lr_gb = LogisticRegression(n_jobs=-1)

    clf_gb.fit(train_x, train_y)
    enc_gb.fit(clf_gb.apply(train_x)[:, :, 0]) #Transformation
    lr_gb.fit(enc_gb.transform(clf_gb.apply(train_x_lr)[:, :, 0]), train_y_lr)

    predictions = lr_gb.predict_proba(
        enc_gb.transform(clf_gb.apply(test_x)[:, :, 0]))[:,1]
    fpr, tpr, _ = roc_curve(test_y, predictions, pos_label=1.0)
    score = round(roc_auc_score(test_y, predictions),4)
    return fpr, tpr, score
