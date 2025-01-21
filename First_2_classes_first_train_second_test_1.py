import pandas as pd
from imblearn.combine import SMOTEENN  # Import SMOTE-ENN
from imblearn.over_sampling import SMOTE, ADASYN
from keras.optimizers import RMSprop
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from sklearn.svm import LinearSVC
# import SVC classifier
from sklearn.svm import SVC
from sklearn import svm
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score # import KFold
import numpy as np
#import sklearn_relief as sr
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

data = pd.read_csv(r'D:\S_H_B_Database_10132022\pyhton codes\pyhton codes\new_train.csv')
data.head()
y = data['SBP-2CLASSES']
X = data.drop(columns=['MIXED-CLASSES'])

#y_train = X['SBP-2CLASSES']
#X=X[['WEIGHT', 'AGE', 'BPM', 'HEIGHT', 'GENDER']]
#X = X.drop(['PAT_ID', 'DBP-2CLASSES', 'SBP-2CLASSES', 'SBP', 'DBP','WEIGHT', 'AGE', 'BPM', 'HEIGHT', 'GENDER'], axis=1)
X = X.drop(['PAT_ID', 'DBP-2CLASSES', 'SBP-2CLASSES', 'SBP', 'DBP'], axis=1)
X_train=X
y_train=y
data_2 = pd.read_csv(r'D:\S_H_B_Database_10132022\pyhton codes\pyhton codes\new_test.csv')
X_other = data_2.drop(columns=['DBP-2CLASSES', 'MIXED-CLASSES', 'SBP-2CLASSES', 'SBP','DBP'], axis=1)
#X_other = data_2.drop(columns=['DBP-2CLASSES', 'MIXED-CLASSES', 'SBP-2CLASSES', 'SBP', 'DBP','WEIGHT', 'AGE', 'BPM', 'HEIGHT', 'GENDER'])
#X_other = data_2[['WEIGHT', 'AGE', 'BPM', 'HEIGHT', 'GENDER','PAT_ID']]
y_other = data_2['SBP-2CLASSES']
# Shuffle and dividing data as train and test for first stage **********************************************************
gs = GroupShuffleSplit(n_splits=2, test_size=.50, random_state=0)
train_ix, test_ix = next(gs.split(X_other, y_other, groups=X_other['PAT_ID']))
X_test = X_other.loc[train_ix]
y_test = y_other.loc[train_ix]
X_val = X_other.loc[test_ix]
y_val = y_other.loc[test_ix]

# X_test_other, y_test_other = shuffle(X_test_other, y_test_other)
X_test = X_test.drop(['PAT_ID'], axis=1)
X_val = X_val.drop(['PAT_ID'], axis=1)

from featurewiz import FeatureWiz
features = FeatureWiz(corr_limit=0.9, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None,
                      verbose=0)
X_train_transformed, important_features = features.fit_transform(X_train, y_train)
X_train_transformed = pd.DataFrame(X_train_transformed).dropna()
# Data Scaling
scaler = StandardScaler()
X_trains = scaler.fit_transform(X_train_transformed)
sm = SMOTE()   #********************************************************************************************************
X_train, y_train = sm.fit_resample(X_trains, y_train)
#adasyn = ADASYN(sampling_strategy={1: 2500, 2: 2500}, n_neighbors=5, random_state=42)
#X_train, y_train = adasyn.fit_resample(X_trains, y_train)

X_test_transformed = features.transform(X_test)
X_val_transformed = features.transform(X_val)

X_test_transformed = pd.DataFrame(X_test_transformed).dropna()
X_val_transformed = pd.DataFrame(X_val_transformed).dropna()

X_test = scaler.transform(X_test_transformed)
X_val = scaler.transform(X_val_transformed)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("X_val shape:", X_val.shape)

# SVM ******************************************************************************************************************
# Hyperparameter tuning for SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['auto', 'scale', 0.1, 0.01],
    'kernel': ['linear', 'rbf', 'poly']
}
# Set probability=True for SVC
grid_search_svm = GridSearchCV(SVC(probability=True), param_grid_svm, cv=5, n_jobs=7)
grid_search_svm.fit(X_train, y_train)
best_svm = grid_search_svm.best_estimator_
print("SVM is OK\n")
# KNN ******************************************************************************************************************
# Hyperparameter tuning for KNN
param_grid_knn = {
    'n_neighbors': [5, 10, 50, 100],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, n_jobs=7)
grid_search_knn.fit(X_train, y_train)
best_knn = grid_search_knn.best_estimator_
print("KNN is OK\n")
# RND ******************************************************************************************************************
# Hyperparameter tuning for Random Forest
param_grid_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, n_jobs=7)
grid_search_rf.fit(X_train, y_train)
best_rnd = grid_search_rf.best_estimator_
print("RND is OK\n")
# Naive Bayes Classifier ***********************************************************************************************
nbc=GaussianNB()
nbc.fit(X_train,y_train)
# Hyperparameter tuning for Logistic Regression
param_grid_log = {
    'C': [0.1, 1, 10, 100],
    'max_iter': [100, 500, 1000]
}
grid_search_log = GridSearchCV(LogisticRegression(), param_grid_log, cv=5, n_jobs=7)
grid_search_log.fit(X_train, y_train)
best_log = grid_search_log.best_estimator_
print("LOG is OK\n")
# Hyperparameter tuning for Decision Tree ******************************************************************************
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5, n_jobs=7)
grid_search_dt.fit(X_train, y_train)
best_dt = grid_search_dt.best_estimator_
print("DT is OK\n")
# Update the Voting Classifier with the best estimators  ***************************************************************
# Update the Voting Classifier with the best estimators and use soft voting  *******************************************
voting = VotingClassifier(
    estimators=[
        ('RND', best_rnd),
        ('DTC', best_dt),
        ('LOG', best_log),
        ('SVC', best_svm),
        ('KNN', best_knn),
        ('NBC', nbc)
    ],
    voting='soft',
    flatten_transform=True  # Allows 'predict_proba' to be used
).fit(X_train, y_train)

# Evaluate models with best hyperparameters  ***************************************************************************
y_pred_voting = voting.predict(X_test)
y_pred_rnd = best_rnd.predict(X_test)
y_pred_svm = best_svm.predict(X_test)
y_pred_knn = best_knn.predict(X_test)
y_pred_nbc = nbc.predict(X_test)
y_pred_log = best_log.predict(X_test)
y_pred_dt = best_dt.predict(X_test)

print('SVM Test accuracy score with default hyperparameters: {0:0.4f}', (classification_report(y_test, y_pred_svm, zero_division=1)))
print('KNN Test accuracy score with default hyperparameters: {0:0.4f}', (classification_report(y_test, y_pred_knn, zero_division=1)))
print('NBC Test accuracy score with default hyperparameters: {0:0.4f}', (classification_report(y_test, y_pred_nbc, zero_division=1)))
print('LOG Test accuracy score with default hyperparameters: {0:0.4f}', (classification_report(y_test, y_pred_log, zero_division=1)))
print('RND Test accuracy score with default hyperparameters: {0:0.4f}', (classification_report(y_test, y_pred_rnd, zero_division=1)))
print('DT Test accuracy score with default hyperparameters: {0:0.4f}', (classification_report(y_test, y_pred_dt, zero_division=1)))
print('VOTING Test accuracy score with default hyperparameters: {0:0.4f}', (classification_report(y_test, y_pred_voting, zero_division=1)))

print('SVM Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_svm)))
print('KNN Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_knn)))
print('NBC Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_nbc)))
print('LOG Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_log)))
print('RND Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_rnd)))
print('DT Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_dt)))
print('VOTING Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_voting)))

# CONf******************************************************************************************************************
cm_svm = confusion_matrix(y_test, y_pred_svm)
print('Confusion Matrix for SVM:')
print(cm_svm)
cm_knn = confusion_matrix(y_test, y_pred_knn)
print('Confusion Matrix for KNN:')
print(cm_knn)
cm_nbc = confusion_matrix(y_test, y_pred_nbc)
print('Confusion Matrix for NBC:')
print(cm_nbc)
cm_log = confusion_matrix(y_test, y_pred_log)
print('Confusion Matrix for LOG:')
print(cm_log)
cm_rnd = confusion_matrix(y_test, y_pred_rnd)
print('Confusion Matrix for RND:')
print(cm_rnd)
cm_dt = confusion_matrix(y_test, y_pred_dt)
print('Confusion Matrix for DT:')
print(cm_dt)
cm_voting = confusion_matrix(y_test, y_pred_voting)
print('Confusion Matrix for VOTING:')
print(cm_voting)

print("Random Forest Best Parameters:", grid_search_rf.best_params_)
print("SVM Best Parameters:", grid_search_svm.best_params_)
print("KNN Best Parameters:", grid_search_knn.best_params_)
print("Logistic Regression Best Parameters:", grid_search_log.best_params_)
print("Decision Tree Best Parameters:", grid_search_dt.best_params_)
print('Voting Test accuracy score with best hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred_voting)))

# For SVM
svm_probs = best_svm.predict_proba(X_test)[:, 1]
auc_svm = roc_auc_score(y_test, svm_probs)
# For KNN
knn_probs = best_knn.predict_proba(X_test)[:, 1]
auc_knn = roc_auc_score(y_test, knn_probs)
# For Naive Bayes (assuming it supports predict_proba)
nbc_probs = nbc.predict_proba(X_test)[:, 1]
auc_nbc = roc_auc_score(y_test, nbc_probs)
# For Logistic Regression
log_probs = best_log.predict_proba(X_test)[:, 1]
auc_log = roc_auc_score(y_test, log_probs)
# For Random Forest
rnd_probs = best_rnd.predict_proba(X_test)[:, 1]
auc_rnd = roc_auc_score(y_test, rnd_probs)
# For Decision Tree
dt_probs = best_dt.predict_proba(X_test)[:, 1]
auc_dt = roc_auc_score(y_test, dt_probs)

# For Voting Classifier
voting_probs = voting.predict_proba(X_test)[:, 1]
auc_voting = roc_auc_score(y_test, voting_probs)

# Plot ROC curve for all classifiers  **********************************************************************************
classifiers = {
    'SVM': (best_svm, svm_probs),
    'KNN': (best_knn, knn_probs),
    'Naive Bayes': (nbc, nbc_probs),
    'Logistic Regression': (best_log, log_probs),
    'Random Forest': (best_rnd, rnd_probs),
    'Decision Tree': (best_dt, dt_probs),
    'Voting Classifier': (voting, voting_probs)
}
plt.figure(figsize=(8, 8))
for name, (classifier, probs) in classifiers.items():
    fpr, tpr, _ = roc_curve(y_test, probs, pos_label=2)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate (B)', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('Receiver Operating Characteristic (ROC) - SMOTE', fontsize=20)
plt.legend()
plt.show()

# Define a function to plot confusion matrix using seaborn  ************************************************************
def plot_cm(ax, clf, X, y_true, title):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)

# Create a figure with subplots  ***************************************************************************************
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Confusion Matrices - ADASYN', fontsize=22)
# Plot confusion matrices for each classifier
plot_cm(axes[0, 0], best_svm, X_test, y_test, 'SVM')
axes[0, 0].set_title('SVM', fontsize=20)
plot_cm(axes[0, 1], best_knn, X_test, y_test, 'KNN')
axes[0, 1].set_title('KNN', fontsize=20)
plot_cm(axes[0, 2], nbc, X_test, y_test, 'Naive Bayes')
axes[0, 2].set_title('Naive Bayes', fontsize=20)
plot_cm(axes[1, 0], best_log, X_test, y_test, 'Logistic Regression')
axes[1, 0].set_title('Logistic Regression', fontsize=20)
plot_cm(axes[1, 1], best_rnd, X_test, y_test, 'Random Forest')
axes[1, 1].set_title('Random Forest', fontsize=20)
plot_cm(axes[1, 2], best_dt, X_test, y_test, 'Decision Tree')
axes[1, 2].set_title('Decision Tree', fontsize=20)
# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
# Display the plot
plt.show()
