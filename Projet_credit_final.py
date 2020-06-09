#!/usr/bin/env python
# coding: utf-8

# # Project on German credit database

# **Project's issue**: 
# 
# The main idea is to challenge the traditional methods of classification and prediction of default with the
# machine-learning methods already used in other financial intermediation institutions, especially in
# startups. The comparison of the methods will be based on the usual performance's metrics. Logistic regression outcomes will be used as benchmark 
#     
# This project consists in building the best possible score function on a classic Machine-learning
# database: the German Credit Data database.
# This database has the distinction of being very small: it represents 1000 loans, 20 variables such as
# marital status, age, number of credits to date, the amount requested. One of the advantages of this
# database is that it contains the same type of information used in default risk assessment in most retail
# banking establishments. However, the defect observation rate is very high (around 30%).
# 
# the Databasae can be find out using the following link :  http://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
# 

# **Main results**
# 
# ![image.png](attachment:image.png)

# # Packages

# In[170]:


import pandas as pd 
import numpy as np 
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,classification_report,auc
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import tree
import tensorflow as tf 
import tensorflow.keras as keras 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import initializers
from tensorflow.keras.utils import plot_model

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.feature_selection import SelectKBest,chi2,RFE,f_classif
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,classification_report,auc
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
import seaborn as sns
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier


# # Load dataset 

# In[44]:


names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount', 
         'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors', 
         'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing', 
         'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']
file = r'D:\sklearn\german.data'
df = pd.read_csv(file,names = names, delimiter = ' ')
df.head()


# In[50]:


df.set_index(np.arange(1,1001,step =1),inplace = True)


# In[52]:


df.describe()


# In[53]:


print("Data set shape", df.shape)


# # Data cleaning 

# In[54]:


# classification variable transformation -> 1 = 'bad' credit; 0 = 'good' credit
df.classification.replace([1,2], [0,1], inplace=True)
# Nombre de 'good' credits  and 'bad credits
df.classification.value_counts()


# In[55]:


#numerical labels
numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age', 
           'existingcredits', 'peopleliable', 'classification']

#categorical labels 
catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
           'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job', 
           'telephone', 'foreignworker'] 


# In[56]:


#standardization of num labels
numdata_std = pd.DataFrame(StandardScaler().fit_transform(df[numvars].drop(['classification'], axis=1)))


# In[57]:


#categorical codification
d = defaultdict(LabelEncoder)
lecatdf = df[catvars].apply(lambda x: d[x.name].fit_transform(x))

# print transformations
for x in range(len(catvars)):
    print(catvars[x],": ", df[catvars[x]].unique())
    print(catvars[x],": ", lecatdf[catvars[x]].unique())


# In[58]:


#One hot encoding, num_var turned into binary variable
dummyvars = pd.get_dummies(df[catvars])


# In[59]:


#clean databse 
data_clean = pd.concat([df[numvars], dummyvars], axis = 1)
print("German credit clean dataset", data_clean.head())
print("Clean dataset set shape", data_clean.shape )


# ## First step features selections : statistical's test
# Features selection based on Pearson correlation , Chi2 and Anova Fvalue

# In[61]:


#Features Default selection
feature_name = data_clean.columns.drop('classification').tolist()
x = data_clean.drop('classification',axis=1)
y = data_clean.classification


# In[62]:


#Preprocessing using Pearson Correlation
cor = data_clean.corr()
#Correlation with output variable
cor_target = abs(cor["classification"])
print(cor_target.describe())


# In[63]:


#Number of highly correlated features with the target value
num_feats = cor_target[cor_target>0.1]
print("Number of highly correlated features with the target value ", len(num_feats))


# In[64]:


num_feats = 15


# In[65]:


def cor_selector(X, y,num_feats):
    cor_list = []
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

def chi_selector(X, y,num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    return chi_support, chi_feature

def fvalue_selector(x, y,num_feats):
    fvalue_selector = SelectKBest(f_classif, k=num_feats)
    fvalue_selector.fit_transform(x, y)
    fvalue_support= fvalue_selector.get_support()
    fvalue_feature = x.loc[:,fvalue_support].columns.tolist()
    return fvalue_support, fvalue_feature


# In[66]:


x


# In[67]:


cor_support, cor_feature = cor_selector(x, y,num_feats)
chi_support, chi_feature = chi_selector(x, y,num_feats)
fvalue_support,fvalue_feature =  fvalue_selector(x, y,num_feats)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support,'Anova Fvalue' :fvalue_support })
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)


# In[69]:


feature_selection_df.head(50)


# In[95]:


relevant_features = feature_selection_df[feature_selection_df.Total>1].Feature
print("Number of relevant features : ", len(relevant_features))
relevant_features


# In[75]:


#Analysis of relevant features correlation between themselves
plt.figure(figsize=(15,15))
cor = x[relevant_features].corr()
columns = np.full((cor.shape[0],), True, dtype=bool)
for i in range(cor.shape[0]):
    for j in range(i+1, cor.shape[0]):
        if abs(cor.iloc[i,j]) >= 0.6:
            if columns[j]:
                columns[j] = False
relevant_features = x[relevant_features].columns[columns]
print("Number of relevant features : ", len(relevant_features))
relevant_features=relevant_features.tolist()
relevant_features


# In[78]:


cor = x[relevant_features].corr()
plt.figure(figsize=(20,20))
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# ## Second step features selection : Wrapper Method
# Recursive features elimination using the Gradient Boosting classifier, Logistic regression and desicion tree

# In[96]:


def rfe_selector(X, y,num_feats,model):
    X_norm = MinMaxScaler().fit_transform(x)
    rfe_selector = RFE(estimator=model, n_features_to_select=num_feats, step=5, verbose=2)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = x.loc[:,rfe_support].columns.tolist()
    return rfe_support, rfe_feature


# In[107]:


x = x[relevant_features]


# In[ ]:


# put all selection together
rfe_selection_df = pd.DataFrame({'Feature':relevant_features, 'Gradient Boosting':xgb_support, 
                                 'Decision tree':tree_support,'Logistic regression' :log_support})
# count the selected times for each feature
rfe_selection_df['Total'] = np.sum(rfe_selection_df, axis=1)
# display the top 100
rfe_selection_df = rfe_selection_df.sort_values(['Total','Feature'] , ascending=False)
rfe_selection_df.index = range(1, len(rfe_selection_df)+1)
rfe_selection_df


# # Modelisation Part 

# ## Function

# In[263]:


def modelfit(model,X_train,Y_train,X_test,Y_test,features,performCV=True,roc=False, printFeatureImportance=False):
    
    #Fitting the model on the data_set
    model.fit(X_train[features],Y_train)
        
    #Predict training and on the test set:
    y_pred1 = model.predict(X_train[features])
    predprob = model.predict_proba(X_train[features])[:,1]
    y_pred2 = model.predict(X_test[features])
    
    # Create and print confusion matrix    
    cfm = confusion_matrix(Y_train,y_pred1)
    cfm_test = confusion_matrix(Y_test,y_pred2)
    print("\nModel Confusion matrix on  train set ")
    print(cfm)
    
    print("\nModel Confusion matrix on  test set ")
    print(cfm_test)
    
    #Print model report:
    print("\nModel Report ")
    print("Accuracy on train set : %.4g" % metrics.accuracy_score(Y_train.values, y_pred1))
    print("Accuracy on test set : %.4g" % metrics.accuracy_score(Y_test.values, y_pred2))
    
    #Perform cross-validation: evaluate using 10-fold cross validation 
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    if performCV:
        print(" cross-validation: evaluate using 10-fold cross validation on the train set ")
        evaluation(model,X_train[features],Y_train,kfold)
    if roc: 
        compute_roc(y_test = Y_test, y_pred_test= y_pred2 ,y_train =Y_train, y_pred_train = y_pred1, plot=True)
          
    #Print Feature Importance:
    if printFeatureImportance:
        feature_importance(model,features,threshold = 0.02, selection=False) 


# In[83]:


#Fonction d'evaluation 
def evaluation(model,X,Y,kfold):
    # Cross Validation to test and anticipate overfitting problem
    scores1 = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    scores2 = cross_val_score(model, X, Y, cv=kfold, scoring='precision')
    scores3 = cross_val_score(model, X, Y, cv=kfold, scoring='recall')
    # The mean score and standard deviation of the score estimate
    print("Cross Validation Accuracy: %0.5f (+/- %0.2f)" % (scores1.mean(), scores1.std()))
    print("Cross Validation Precision: %0.5f (+/- %0.2f)" % (scores2.mean(), scores2.std()))
    print("Cross Validation Recall: %0.5f (+/- %0.2f)" % (scores3.mean(), scores3.std()))
    return  


# In[86]:


def compute_roc(y_test, y_pred_test, y_train, y_pred_train, plot=True):
    fpr_train = dict()
    tpr_train = dict()
    auc_score_train = dict()   
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
    auc_score_train = auc(fpr_train, tpr_train)
    
    fpr_test = dict()
    tpr_test = dict()
    auc_score_test = dict()
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test)
    auc_score_test = auc(fpr_test, tpr_test)
    
    if plot:
        plt.figure(figsize=(7, 6))
        
        plt.plot(fpr_train, tpr_train, color='blue',
                 label='ROC curve Train data set (area = %0.2f)' % auc_score_train)
        
        plt.plot(fpr_test, tpr_test, color='orange',
                 label='ROC curve Test data set (area = %0.2f)' % auc_score_test)
        
        plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        
        plt.legend(loc="upper right")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title("ROC Curve")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.show()
    return 


# In[150]:


def feature_importance(model,features,threshold,selection=False) : 
    feature_importances = pd.DataFrame(model.feature_importances_  )
    feature_importances = feature_importances.T
    feature_importances.columns = [features]
    
    sns.set(rc={'figure.figsize':(13,12)})
    fig = sns.barplot(data=feature_importances, orient='h', order=feature_importances.mean().sort_values(ascending=False).index)
    fig.set(title = 'Feature importance', xlabel = 'features', ylabel = 'features_importance' )
    
    if selection: #Selection of features with min threshold% of feature importance
        n_features = feature_importances[feature_importances.loc[:,] > threshold].dropna(axis='columns')
        n_features = n_features.columns.get_level_values(0)    
        print("Selected features")
        print(n_features)
        
    return fig


# ## Local train and test dataset

# In[85]:


relevant_features = ['savings_A65',
 'savings_A61',
 'purpose_A43',
 'property_A124',
 'property_A121',
 'existingchecking_A14',
 'existingchecking_A12',
 'existingchecking_A11',
 'credithistory_A34',
 'credithistory_A31',
 'credithistory_A30',
 'otherinstallmentplans_A143',
 'housing_A152',
 'duration']


# In[10]:


#Train set validation set 
x= data_clean.drop('classification', axis=1)
y = data_clean['classification']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# # Classification algorithms

# ## ML DecisionTreeClassifier

# In[87]:


#Decison tree baseline model
model = tree.DecisionTreeClassifier()
#Fitting Decison tree baseline model
modelfit(model,x_train, y_train,relevant_features,performCV=False)
print("Accuracy on test set :{:.3f} ".format(model.score(x_test, y_test)))


# In[88]:


#Tunning Decision tree model  With Gridsearch
print('Decision tree with Classifier')
params={'max_depth': np.arange(2, 8),'criterion':['gini','entropy']}
tree_estimator = tree.DecisionTreeClassifier()

kfold = 10 

grid_tree = GridSearchCV(tree_estimator, params, cv=kfold, scoring="accuracy",
                         n_jobs=1,
                         verbose=False)

grid_tree.fit(x_train, y_train)
best_est = grid_tree.best_estimator_
print(best_est)
print(grid_tree.best_score_)


# summarize results
print("Best: %f using %s" % (grid_tree.best_score_, grid_tree.best_params_))
means = grid_tree.cv_results_['mean_test_score']
stds = grid_tree.cv_results_['std_test_score']
params = grid_tree.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#  **the best Hyperparameters for our Decision tree model using gridsearch Cv  is {'criterion': 'gini', 'max_depth': 5}**

# In[91]:


model = tree.DecisionTreeClassifier(max_depth = 5,criterion='gini')
modelfit(model,x_train, y_train,relevant_features,printFeatureImportance=True)


# In[93]:


feature_importance(model,relevant_features,0.05,selection=True)


# In[134]:


model = tree.DecisionTreeClassifier(max_depth = 5,criterion='gini')
tree_features = ['savings_A65', 'existingchecking_A14', 'existingchecking_A11',
                 'credithistory_A34', 'credithistory_A30', 'otherinstallmentplans_A143','duration']
cn=['0', '1']
clf = model.fit(x_train[tree_features], y_train)
plot_tree(clf, 
          feature_names =tree_features,
          class_names = cn,
          filled=True,fontsize=10)
plt.show()


# In[143]:


modelfit(model,x_train,y_train,tree_features, printFeatureImportance=False)
modelfit(model,x_test,y_test,tree_features, printFeatureImportance=False)
y_pred = model.predict(x_train[tree_features])
y_pred2 = model.predict(x_test[tree_features])
compute_roc(y_test = y_test, y_pred_test= y_pred2 ,y_train =y_train, y_pred_train = y_pred, plot=True)


# ## ML Gradient Boosting classifier

# In[227]:


#Train set validation set 
x= data_clean.drop('classification', axis=1)
y = data_clean['classification']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[169]:


GradientBoostingClassifier(random_state=10)


# In[151]:


#Baseline Gradient boosting model 
base_gbm = GradientBoostingClassifier(random_state=10)
modelfit(base_gbm,x_train, y_train,relevant_features,roc=True,printFeatureImportance=True)


# # **Tunning parameters with Gridsearch**
# ** Baseline approch**
#    *Fix learning rate and number of estimators for tuning tree-based parameters
#     min_samples_split = 500 : This should be ~0.5-1% of total values.
#     min_samples_leaf = 50 :  for preventing overfitting and again a small value.
#     max_depth = 8 : Should be choosen (5-8) based on the number of observations and predictors.
#     max_features = ‘sqrt’ : Its a general thumb-rule to start with square root.
#     subsample = 0.8 : commonly used used start value
# 
# **we will choose all the features 

# In[152]:


print('tuning n_estimators')
params1 = {'n_estimators':range(30,81,10)}

estimator = GradientBoostingClassifier(learning_rate=0.1, 
                                       min_samples_split=500,
                                       min_samples_leaf=50,
                                       max_depth=8,
                                       max_features='sqrt',
                                       subsample=0.8,
                                       random_state=10)

grid_xgb1 = GridSearchCV(estimator,
                  params1,
                  cv=10,
                  scoring='accuracy',
                  n_jobs=1,
                  verbose=False)

grid_result=grid_xgb1.fit(x_train[relevant_features], y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[153]:


print('tuning max_depth and min_sample_split')
params2 =  {'max_depth':range(5,16,2), 'min_samples_split':range(400,1001,200)}

estimator = GradientBoostingClassifier(learning_rate=0.1,
                                       n_estimators = 80,
                                       max_features='sqrt',
                                       subsample=0.8,
                                       random_state=10)

grid_xgb2 = GridSearchCV(estimator,
                  params2,
                  cv=10,
                  scoring='accuracy',
                  n_jobs=-1,
                  verbose=True)

grid_result=grid_xgb2.fit(x_train[relevant_features], y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[154]:


print('tuning num_sample_split and min_sample_split')
params3 =  {'min_samples_leaf':range(40,70,10), 'min_samples_split':range(400,1001,200)}
estimator = GradientBoostingClassifier(learning_rate=0.1,
                                       n_estimators = 80,
                                       max_depth=5,
                                       max_features='sqrt',
                                       subsample=0.8,
                                       random_state=10)
grid_xgb3 = GridSearchCV(estimator,
                  params3,
                  cv=10,
                  scoring='accuracy',
                  n_jobs=-1,
                  verbose=True)
grid_result=grid_xgb3.fit(x_train[relevant_features], y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[157]:


print('tuning max_features')
params4 =  {'max_features':range(2,len(relevant_features),1)}

estimator = GradientBoostingClassifier(learning_rate=0.1,
                                       n_estimators = 40,
                                       max_depth=5,                                        
                                       min_samples_split=400, 
                                       min_samples_leaf=40, 
                                       subsample=0.8,
                                       random_state=10)
grid_xgb4 = GridSearchCV(estimator,
                  params4,
                  cv=10,
                  scoring='accuracy',
                  n_jobs=1,
                  verbose=True)
grid_result=grid_xgb4.fit(x_train[relevant_features], y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds , params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[265]:


xgb_tunned = GradientBoostingClassifier(learning_rate=0.1,
                                        n_estimators = 40,
                                        max_depth=5,                         
                                        min_samples_split=400,
                                        min_samples_leaf=40,
                                        subsample=0.8,
                                        max_features= 11,
                                        random_state=10)


# In[266]:


#Fit Cross validation and prediction on the train and the test set
modelfit(xgb_tunned,x_train, y_train, x_test,y_test,relevant_features,performCV=True,roc=True,printFeatureImportance=True)


# ## MLP MODEL AND LOGISTIC REGRESSION 

# In[75]:


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
    
    print ("Logistic model Accuracy: %.6f" %accuracy_score(y_test,y_pred) )
    
    # Create and print confusion matrix
    print("ROC curve")
    compute_roc(y_test, y_pred, plot=True)
    
    # Get Cross Validation and Confusion matrix on the test dataset
    evaluation(model, x_test, y_test)
    return 


# # Tunning Hyperparameters with Gridsearch

# In[14]:


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
                    scoring='roc_auc',
                    verbose=False)

grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# # Model implementation

# In[69]:


# Logistic Regression
logreg(x_train, y_train,x_test,y_test)


# In[72]:


#Decison tree baseline model
model = tree.DecisionTreeClassifier()
#Fitting Decison tree baseline model
model.fit(x_train,y_train)
#Predict target variables y for test data
y_pred = model.predict(x_test) 
#Create and print confusion matrix
print("ROC curve")
compute_roc(y_test, y_pred, plot=True)
cfm = confusion_matrix(y_test,y_pred)
print ("Decision tree model Accuracy: %.6f" %accuracy_score(y_test,y_pred))
print("confusion matrix")
print(cfm)
evaluation(model,x_test,y_test)


# In[73]:


model = tree.DecisionTreeClassifier(max_depth = 3)
#Fitting Decison tree baseline model
model.fit(x_train,y_train)
# Predict target variables y for test data
y_pred = model.predict(x_test) 
# Create and print confusion matrix
print("ROC curve")
print ("Decison tree model Accuracy: %.6f" %accuracy_score(y_test,y_pred) )
compute_roc(y_test, y_pred, plot=True)
cfm = confusion_matrix(y_test,y_pred)
print ("Decision tree model Accuracy: %.6f" %accuracy_score(y_test,y_pred))
print("confusion matrix")
print(cfm)
evaluation(model,x_test,y_test)


# In[60]:


#MLP implementation 
best_pipe = grid_result.best_estimator_
print('Scorer:' , grid_result.scorer_)
print('Best_score: %.5f' % grid_result.best_score_)

# Fitting the best pipe model 
history=best_pipe.fit(x_train, y_train, verbose =2 , validation_data=(x_test, y_test))

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#print(y_pred)
score = best_pipe.score(x_test, y_test)
print('Score on the test set: %.5f' %  score)
print('Cross validation of Gridsearch best estimator')
evaluation(best_pipe,x_test,y_test)

y_pred = best_pipe.predict(x_train)
y_pred2 = best_pipe.predict(x_test)

compute_roc(y_test = y_test, y_pred_test= y_pred2 ,y_train =y_train, y_pred_train = y_pred, plot=True)


# In[ ]:


{'batch_size': 100, 'epochs': 15, 'init': 'glorot_uniform', 'learning_rate': 0.01}


# In[77]:


model = mlp_model(init = 'uniform',learning_rate=0.01)

# Fitting the best pipe model 
history= model.fit(x_train, y_train, batch_size=100,epochs= 15,verbose =2,validation_data=(x_test, y_test))

plot_model(model,to_file='model.png')

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Predict on test set and on train set

y_pred = model.predict(x_train)
y_pred2 = model.predict(x_test)

#print('Score on the train set: %.5f' %  model.score(x_test, y_test))
#print('Score on the test set: %.5f' %  model.score(x_train, y_train))

#print('Cross validation of Gridsearch best estimator')
#evaluation(model,x_test,y_test)

compute_roc(y_test = y_test, y_pred_test= y_pred2 ,y_train =y_train, y_pred_train = y_pred, plot=True)

