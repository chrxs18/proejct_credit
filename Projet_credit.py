#!/usr/bin/env python
# coding: utf-8

# # Project on German credit

# # Packages

# In[112]:


import pandas as pd 
import numpy as np 
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,classification_report,auc
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from xgboost import XGBClassifier

import tensorflow as tf 
import tensorflow.keras as keras 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import initializers


# # Load dataset 

# In[3]:



names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount', 
         'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors', 
         'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing', 
         'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

file = r'D:\sklearn\german.data'
df = pd.read_csv(file,names = names, delimiter = ' ')
print("German Credit Data set", df.head() )
print("Data set shape", df.shape )


# # Data Analysis

# In[ ]:





# # Data cleaning 

# In[4]:


# Transformation de la variable classification -> 0 = 'bad' credit; 1 = 'good' credit
df.classification.replace([1,2], [1,0], inplace=True)
# Nombre de 'good' credits  and 'bad credits
df.classification.value_counts()


# In[6]:


#Variables nurmeriques 
numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age', 
           'existingcredits', 'peopleliable', 'classification']

#Variables categorielles 
catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
           'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job', 
           'telephone', 'foreignworker'] 


# In[7]:


#Centration-reduction des variables numeriques 
numdata_std = pd.DataFrame(StandardScaler().fit_transform(df[numvars].drop(['classification'], axis=1)))


# In[8]:


#Codification des variables categorielles 
d = defaultdict(LabelEncoder)
lecatdf = df[catvars].apply(lambda x: d[x.name].fit_transform(x))

# print transformations
for x in range(len(catvars)):
    print(catvars[x],": ", df[catvars[x]].unique())
    print(catvars[x],": ", lecatdf[catvars[x]].unique())


# In[9]:


#One hot encoding, creation d'une variable binanire pour chaque variable categorielle 
dummyvars = pd.get_dummies(df[catvars])


# In[10]:


#Creation de la clean database 
data_clean = pd.concat([df[numvars], dummyvars], axis = 1)
print("German credit clean dataset", data_clean.head())
print("Clean dataset set shape", data_clean.shape )


# # Train and test dataset

# In[11]:


#Train set validation set 
x= data_clean.drop('classification', axis=1)
y = data_clean['classification']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
print("x_train", x_train.shape)
print("x_test", x_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)
y_test.value_counts()
y_train.value_counts()


# # Classification algorithms

# # Functions

# In[31]:


#Fonction d'evaluation 
def evaluation(model,X,Y):
   # Cross Validation to test and anticipate overfitting problem
   scores1 = cross_val_score(model, X, Y, cv=2, scoring='accuracy')
   scores2 = cross_val_score(model, X, Y, cv=2, scoring='precision')
   scores3 = cross_val_score(model, X, Y, cv=2, scoring='recall')
   scores4 = cross_val_score(model, X, Y, cv=2, scoring='roc_auc')
   # The mean score and standard deviation of the score estimate
   print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std()))
   print("Cross Validation Precision: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std()))
   print("Cross Validation Recall: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std()))
   print("Cross Validation roc_auc: %0.2f (+/- %0.2f)" % (scores4.mean(), scores4.std()))
   return  


# In[107]:


def compute_roc(y_true, y_pred, plot=True):
    fpr = dict()
    tpr = dict()
    auc_score = dict()
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        plt.legend(loc="upper right")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title("ROC Curve")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.show()
    return fpr, tpr, auc_score


# In[114]:


#Baseline mlp model 
def mlp_model(init,learning_rate):   
    
    # Model initialisation and layers specification
    model = tf.keras.Sequential()
    model.add(Dense(61,input_dim=61,kernel_initializer=init, activation = 'relu'))
    model.add(Dense(1,kernel_initializer=init, activation='sigmoid'))
    
    # Initialisation of the optimizer
    adam = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile model 
    model.compile(loss=binary_crossentropy, optimizer=adam, metrics=['accuracy']) 
    return model

# Gradient boosting model
def xgboost(params, X_train, y_train,X_test, y_test):
    print('XGBoost v',xgb.__version__)
    eval_set=[(X_train, y_train), (X_test, y_test)]
    
    model = XGBClassifier(**params).      fit(X_train, y_train, eval_set=eval_set,                   eval_metric='auc', early_stopping_rounds = 100, verbose=100)
        
    print(model.best_ntree_limit)

    model.set_params(**{'n_estimators': model.best_ntree_limit})
    model.fit(X_train, y_train)
    print(model,'\n')
    
    # Predict target variables y for test data
    y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit) 
    print(y_pred)
   
    #Get Cross Validation and Confusion matrix
    evaluation(model, X_train, y_train)
    
    
    # Create and print confusion matrix    
    cfm = confusion_matrix(y_test,y_pred)
    print(cfm)
    
    print (classification_report(y_test,y_pred) )
    print ('\n')
    print ("XGBoost model Accuracy: %.6f" %accuracy_score(y_test,y_pred) )
    
    # ROC Curve
    print("ROC curve")
    compute_roc(y_test, y_pred, plot=True)
    
    return model



# fit, train and cross validate Logisitc regression with training and test data 
def logreg(X_train, y_train,X_test, y_test):
    print("LogisticRegression")
    model = LogisticRegression().fit(X_train, y_train)
    print(model,'\n')
    
    # Predict target variables y for test data
    y_pred = model.predict(X_test)
    
    print (classification_report(y_test,y_pred) )
    
    # Create and print confusion matrix 
    print("confusion matrix")
    cfm = confusion_matrix(y_test,y_pred)
    print(cfm)
    
    # Create and print confusion matrix
    print("ROC curve")
    compute_roc(y_test, y_pred, plot=True)
    
    # Get Cross Validation and Confusion matrix on the test dataset
    evaluation(model, x_test, y_test)
    return 


# In[113]:


# Logistic Regression
logreg(x_train, y_train,x_test,y_test)


# # Tunning Hyperparameters with Gridsearch

# In[72]:


print('MLP with grid search')
# Model creation with keras wrapper
estimator = KerasClassifier(build_fn=mlp_model,verbose=0)

# grid search epochs, batch size, learning_rate, initialiazer
batches = [100]
lr = [ 0.01, 0.001]
init = ['normal', 'uniform','glorot_uniform']
epochs = [10, 15]
param_grid = dict(epochs=epochs, learning_rate=lr, batch_size = batches, init = init)
grid = GridSearchCV(estimator, param_grid=param_grid,
                    return_train_score=False,
                    scoring=['accuracy', 'roc_auc'],
                    refit= 'accuracy')

grid_result = grid.fit(x_train, y_train)


# In[73]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_roc_auc']
stds = grid_result.cv_results_['std_test_roc_auc']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[79]:


print('XGBoost with grid search')
params={
    'learning_rate': [0.01, 0.02],
    'max_depth': [3],
    #'subsample': [0.6]
    #'colsample_bytree': [0.5],
    'n_estimators': [50, 100, 200, 300, 400, 500]
    #'reg_alpha': [0.03] #[0.01, 0.02, 0.03, 0.04]
}


estimator = xgb.XGBClassifier()

grid_xgb = GridSearchCV(estimator,
                  params,
                  cv=2,
                  scoring="roc_auc",
                  n_jobs=1,
                  verbose=False)
grid_xgb.fit(x_train, y_train)
best_est = grid_xgb.best_estimator_
print(best_est)
print(grid_xgb.best_score_)

print('Cross validation of Gridsearch best estimator')
evaluation(best_est,x_test,y_test)


# # Model implementation

# In[106]:


# Logistic Regression implmentation
logreg(x_train, y_train,x_test,y_test)


# In[70]:


#MLP implementation 

best_pipe = grid_result.best_estimator_
print('Scorer:' , grid_result.scorer_)
print('Best_score: %.5f' % grid_result.best_score_)
#y_pred = best_pipe.predict(x_test)
#print(y_pred)
score = best_pipe.score(x_test, y_test)
print('Score on the test set: %.5f' % score)

print('Cross validation of Gridsearch best estimator')
evaluation(best_pipe,x_test,y_test)


# In[115]:


#Xgboost implementation 
#Baseline model
params={}
xgboost(params, x_train, y_train, x_test, y_test)


# In[55]:


# Train xgboost
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test, y_test)
param = {'max_depth' : 3, 'eta' : 0.1, 'objective' : 'binary:logistic', 'seed' : 42}
num_round = 50
bst = xgb.train(param, dtrain, num_round, [(dtest, 'test'), (dtrain, 'train')])


# In[56]:


preds = bst.predict(dtest)
preds[preds > 0.5] = 1
preds[preds <= 0.5] = 0
print(accuracy_score(preds, y_test), 1 - accuracy_score(preds, y_test))

