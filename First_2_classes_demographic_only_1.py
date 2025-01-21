import export
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from featurewiz import FeatureWiz
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import sem, t
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor, RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import os
import shap
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.preprocessing import PolynomialFeatures
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from imblearn.over_sampling import SMOTE, ADASYN

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print("CUDA Support:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Active GPU:", torch.cuda.get_device_name(0))
    print("GPU Count:", torch.cuda.device_count())
else:
    print("CUDA not supported or no any GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def confidence_interval(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, h
def stratified_group_split(X, y, groups, test_size=0.3, val_size=0.3, random_state=42, return_group_ids=False):
    """
    Stratifikasyona dayalı grup bölme işlemi:
    - Sınıflar arasında eşit dağılım sağlar.
    - PAT_ID gibi grupların bölünmesini garantiler.

    Parameters:
    - X: Özellikler
    - y: Hedef değişken
    - groups: Grup bilgileri (ör. PAT_ID)
    - test_size: Test seti oranı
    - val_size: Validation seti oranı
    - random_state: Rastgelelik kontrolü için
    - return_group_ids: Grupları (ör. PAT_ID) geri döndürme seçeneği

    Returns:
    - Eğitim, doğrulama ve test setleri (X_train, y_train, X_val, y_val, X_test, y_test)
    - (Opsiyonel) Eğitim, doğrulama ve test grupları (train_groups, val_groups, test_groups)
    """
    # PAT_ID'lere göre unique grupları ve sınıfları belirle
    unique_groups = pd.DataFrame({"group": groups, "y": y}).drop_duplicates()
    class_group_map = unique_groups.groupby("y")["group"].apply(list).to_dict()
    train_groups, val_groups, test_groups = [], [], []

    for cls, grp_list in class_group_map.items():
        # Grupları karıştır
        grp_list = shuffle(grp_list, random_state=random_state)
        # İlk bölme: Train+Validation ve Test
        grp_train_val, grp_test = train_test_split(
            grp_list, test_size=test_size, random_state=random_state
        )
        # İkinci bölme: Train ve Validation
        grp_train, grp_val = train_test_split(
            grp_train_val, test_size=val_size, random_state=random_state
        )
        train_groups.extend(grp_train)
        val_groups.extend(grp_val)
        test_groups.extend(grp_test)

    # Gruplara göre veriyi maskele
    groups = pd.Series(groups)
    train_mask = groups.isin(train_groups)
    val_mask = groups.isin(val_groups)
    test_mask = groups.isin(test_groups)

    # Ana dönüş
    result = (
        X[train_mask], y[train_mask],  # Eğitim
        X[val_mask], y[val_mask],  # Doğrulama
        X[test_mask], y[test_mask],  # Test
    )

    # Opsiyonel: Grup ID'lerini de döndür
    if return_group_ids:
        return result + (
            groups[train_mask],  # Eğitim grupları
            groups[val_mask],    # Doğrulama grupları
            groups[test_mask],   # Test grupları
        )

    return result
def augment_with_interpolation(X, y, num_augmentations=5):
    augmented_X = []
    augmented_y = []

    for i in range(len(X)):
        for _ in range(num_augmentations):
            # Rastgele bir diğer örnek seç
            idx2 = np.random.choice(len(X), size=1, replace=False)[0]

            # Rastgele ağırlık
            alpha = np.random.uniform(0, 1)

            # Yeni örnek oluştur
            new_sample_X = alpha * X[i] + (1 - alpha) * X[idx2]
            new_sample_y = alpha * y[i] + (1 - alpha) * y[idx2]

            augmented_X.append(new_sample_X)
            augmented_y.append(new_sample_y)

    # Yeni verileri numpy array olarak döndür
    return np.array(augmented_X), np.array(augmented_y)
def apply_smote(X_train, y_train):
    print("SMOTE Öncesi Sınıf Dağılımı:", Counter(y_train))
    # smote = SMOTE(sampling_strategy='auto', k_neighbors=5, n_jobs=1)  # n_jobs=1: Tek iş parçacığı kullanır
    # X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    ros = RandomOverSampler(random_state=42)
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)
    print("SMOTE Sonrası Sınıf Dağılımı:", Counter(y_train_balanced))
    return X_train_balanced, y_train_balanced
def summarize_class_distribution(y, group_ids, group_name):
    summary = pd.DataFrame({"PAT_ID": group_ids, "y": y}).groupby("PAT_ID")["y"].value_counts().unstack(
        fill_value=0)
    print(f"\n{group_name} Class Distribution Per PAT_ID:")
    print(summary)
    print(f"\n{group_name} Total Class Counts:")
    print(summary.sum())
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
def stratified_group_kfold_old(X, y, groups, n_splits=5, random_state=42):
    """
    Stratifikasyona dayalı grup bazlı 5-Fold Cross-Validation:
    - Sınıflar arasında eşit dağılım sağlar.
    - Grupların (ör. PAT_ID) bölünmesini garantiler.
    Parameters:
    - X: Özellikler (DataFrame veya NumPy array)
    - y: Hedef değişken (Liste veya NumPy array)
    - groups: Grup bilgileri (ör. PAT_ID)
    - n_splits: Fold sayısı (Varsayılan: 5)
    - random_state: Rastgelelik kontrolü için
    Returns:
    - Her bir fold için (%60 train, %10 test, %30 val):
      train, validation, test setleri ve ilgili grup ID'leri
    """
    unique_groups = pd.DataFrame({"group": groups, "y": y}).drop_duplicates()
    class_group_map = unique_groups.groupby("y")["group"].apply(list).to_dict()
    kfold_results = []  # Sonuçları burada saklayacağız
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(unique_groups["group"], unique_groups["y"])):
        test_groups = unique_groups.iloc[test_idx]["group"].tolist()
        train_val_groups = unique_groups.iloc[train_val_idx]["group"].tolist()
        train_groups, val_groups = [], []
        # Validation ve Train ayrımı için iç stratifikasyon
        for cls, grp_list in class_group_map.items():
            grp_list = [g for g in grp_list if g in train_val_groups]  # Bu fold'un grup subsetine indir
            grp_list = shuffle(grp_list, random_state=random_state + fold)

            grp_train, grp_val = train_test_split(
                grp_list, test_size=0.3, random_state=random_state + fold
            )
            train_groups.extend(grp_train)
            val_groups.extend(grp_val)
        # Gruplara göre veriyi maskele
        groups_series = pd.Series(groups)
        train_mask = groups_series.isin(train_groups)
        val_mask = groups_series.isin(val_groups)
        test_mask = groups_series.isin(test_groups)
        # Fold sonuçlarını sakla
        kfold_results.append(
            (
                X[train_mask], y[train_mask],  # Eğitim seti
                X[val_mask], y[val_mask],  # Doğrulama seti
                X[test_mask], y[test_mask],  # Test seti
                groups_series[train_mask],  # Eğitim grupları
                groups_series[val_mask],  # Doğrulama grupları
                groups_series[test_mask],  # Test grupları
            )
        )
    return kfold_results
def group_stratified_kfold(X, y, groups, n_splits=5, random_state=42):
    """
    Grup bazlı Stratifikasyona uygun 5-Fold Cross-Validation:
    - Her fold için gruplar, yalnızca bir set içinde (train, validation veya test) bulunur.
    - Grup bazlı bölme yapılır ve sınıf dengesi korunur.

    Parameters:
    - X: Özellikler (DataFrame veya NumPy array)
    - y: Hedef değişken (Liste veya NumPy array)
    - groups: Grup bilgileri (ör. PAT_ID)
    - n_splits: Fold sayısı (Varsayılan: 5)
    - random_state: Rastgelelik kontrolü için

    Returns:
    - Her fold için:
      train, validation, test setleri ve ilgili grup ID'leri
    """
    # Benzersiz gruplar ve hedef değişkenleri ilişkilendir
    unique_groups = pd.DataFrame({"group": groups, "y": y}).drop_duplicates()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kfold_results = []

    for train_val_idx, test_idx in skf.split(unique_groups["group"], unique_groups["y"]):
        # Test gruplarını belirle
        test_groups = unique_groups.iloc[test_idx]["group"].tolist()
        train_val_groups = unique_groups.iloc[train_val_idx]["group"].tolist()

        # Train/Validation ayrımı
        train_val_split = StratifiedKFold(n_splits=n_splits - 1, shuffle=True, random_state=random_state)
        for train_idx, val_idx in train_val_split.split(train_val_groups, unique_groups.iloc[train_val_idx]["y"]):
            train_groups = [train_val_groups[i] for i in train_idx]
            val_groups = [train_val_groups[i] for i in val_idx]

            # Maskeleri oluştur
            groups_series = pd.Series(groups)
            train_mask = groups_series.isin(train_groups)
            val_mask = groups_series.isin(val_groups)
            test_mask = groups_series.isin(test_groups)

            # Fold sonuçlarını sakla
            kfold_results.append(
                (
                    X[train_mask], y[train_mask],  # Eğitim seti
                    X[val_mask], y[val_mask],  # Doğrulama seti
                    X[test_mask], y[test_mask],  # Test seti
                    groups_series[train_mask],  # Eğitim grupları
                    groups_series[val_mask],  # Doğrulama grupları
                    groups_series[test_mask],  # Test grupları
                )
            )
            break  # Train/Val split'ten sadece bir set al
    return kfold_results


data = pd.read_csv(r'D:\S_H_B_Database_10132022\pyhton codes\pyhton codes\new_train.csv')
y = data['DBP-2CLASSES']
#X = data.drop(columns=['MIXED-CLASSES','DBP-2CLASSES','SBP-2CLASSES','SBP',
 #                      'DBP','GENDER','AGE','WEIGHT','HEIGHT','BPM'], axis=1)
X = data[['GENDER','AGE','WEIGHT','HEIGHT','PAT_ID']]
# Apply SMOTE to training data only
#sm = SMOTE()
#X_train_resampled, y_train_resampled = sm.fit_resample(X, y)
# Create a GroupShuffleSplit object
group_split = GroupShuffleSplit(n_splits=2, test_size=0.3, random_state=42)
# Define the features and target variables
features1 = X.drop('PAT_ID', axis=1)  # Assuming 'PAT_ID' is the column containing group information
target = y
# Generate the train/validation/test indices
train_idx, val_test_idx = next(group_split.split(features1, target, groups=X['PAT_ID']))
val_idx, test_idx = next(group_split.split(features1.iloc[val_test_idx], target.iloc[val_test_idx],
                                           groups=X.iloc[val_test_idx]['PAT_ID']))
# Split the data into train, validation, and test sets
X_train, y_train = features1.iloc[train_idx], target.iloc[train_idx]
X_val, y_val = features1.iloc[val_idx], target.iloc[val_idx]
X_test, y_test = features1.iloc[test_idx], target.iloc[test_idx]
# Print the shapes
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_val.shape, y_val.shape)

#from featurewiz import FeatureWiz
#features = FeatureWiz(corr_limit=0.9, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None,
 #                     verbose=1)
#X_train_transformed, important_features = features.fit_transform(X_train, y_train)
#X_train_transformed = pd.DataFrame(X_train_transformed).dropna()
# Data scaling
scaler = StandardScaler()
X_trains = scaler.fit_transform(X_train)
sm = SMOTE()   #***SMOTE ***********************************************************************************************
X_train, y_train = sm.fit_resample(X_trains, y_train)
#adasyn = ADASYN(sampling_strategy={1: 2500, 2: 2500}, n_neighbors=5, random_state=42)
#X_train, y_train = adasyn.fit_resample(X_trains, y_train)
# Test ve validasyon setini aynı işlemleri geçirebilirsiniz
#X_test_transformed = features.transform(X_test)
#X_val_transformed = features.transform(X_val)
#X_test_transformed = pd.DataFrame(X_test_transformed).dropna()
#X_val_transformed = pd.DataFrame(X_val_transformed).dropna()
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
# Size control
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
# RF *******************************************************************************************************************
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
nbc = GaussianNB()
nbc.fit(X_train, y_train)
# Hyperparameter tuning for Logistic Regression ************************************************************************
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
# Update the Voting Classifier with the best estimators ****************************************************************
# Update the Voting Classifier with the best estimators and use soft voting ********************************************
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

# Evaluate models with best hyperparameters ****************************************************************************
y_pred_voting = voting.predict(X_test)
y_pred_rnd = best_rnd.predict(X_test)
y_pred_svm = best_svm.predict(X_test)
y_pred_knn = best_knn.predict(X_test)
y_pred_nbc = nbc.predict(X_test)
y_pred_log = best_log.predict(X_test)
y_pred_dt = best_dt.predict(X_test)

print('SVM Test accuracy score with default hyperparameters: {0:0.4f}',
      (classification_report(y_test, y_pred_svm, zero_division=1)))
print('KNN Test accuracy score with default hyperparameters: {0:0.4f}',
      (classification_report(y_test, y_pred_knn, zero_division=1)))
print('NBC Test accuracy score with default hyperparameters: {0:0.4f}',
      (classification_report(y_test, y_pred_nbc, zero_division=1)))
print('LOG Test accuracy score with default hyperparameters: {0:0.4f}',
      (classification_report(y_test, y_pred_log, zero_division=1)))
print('RND Test accuracy score with default hyperparameters: {0:0.4f}',
      (classification_report(y_test, y_pred_rnd, zero_division=1)))
print('DT Test accuracy score with default hyperparameters: {0:0.4f}',
      (classification_report(y_test, y_pred_dt, zero_division=1)))
print('VOTING Test accuracy score with default hyperparameters: {0:0.4f}',
      (classification_report(y_test, y_pred_voting, zero_division=1)))

print('SVM Test accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred_svm)))
print('KNN Test accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred_knn)))
print('NBC Test accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred_nbc)))
print('LOG Test accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred_log)))
print('RND Test accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred_rnd)))
print('DT Test accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred_dt)))
print('VOTING Test accuracy score with default hyperparameters: {0:0.4f}'.format(
    accuracy_score(y_test, y_pred_voting)))
#****CONFUSION MATRIX **************************************************************************************************
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
print(
    'Voting Test accuracy score with best hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred_voting)))
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
# Plot ROC curve for all classifiers ***********************************************************************************
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

# Define a function to plot confusion matrix using seaborn *************************************************************
def plot_cm(ax, clf, X, y_true, title):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)

# Create a figure with subplots  ***************************************************************************************
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Confusion Matrices - SMOTE', fontsize=22)
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
