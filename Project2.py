import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

df = pd.read_csv("pollution.csv")

label_encoder = LabelEncoder()
df['Air Quality'] = label_encoder.fit_transform(df['Air Quality'])

df.fillna(df.mean(), inplace=True)

X = df.drop("Air Quality", axis=1)
y = df["Air Quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)
X_resampled_pca = pca.fit_transform(X_resampled_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original number of features: {X_resampled.shape[1]}")
print(f"Number of features after PCA: {X_resampled_pca.shape[1]}")

print("Before SMOTE class distribution:")
print(y_train.value_counts())

print("After SMOTE class distribution:")
print(y_resampled.value_counts())


# Worked done by umer
# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

rf = RandomForestClassifier(random_state=42)

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_rf.fit(X_resampled_pca, y_resampled)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test_pca)
print("Best Parameters (GridSearchCV):", grid_rf.best_params_)
print("Random Forest Accuracy (GridSearchCV):", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

conf_matrix = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=best_rf.classes_, yticklabels=best_rf.classes_)
plt.title("Confusion Matrix (Random Forest)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


from sklearn.model_selection import RandomizedSearchCV

param_dist_rf = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_dist_rf, n_iter=50, cv=5, 
                               n_jobs=-1, verbose=2, random_state=42)
random_rf.fit(X_resampled_pca, y_resampled)

best_rf_random = random_rf.best_estimator_
y_pred_rf_random = best_rf_random.predict(X_test_pca)
print("Best Parameters (RandomizedSearchCV):", random_rf.best_params_)
print("Random Forest Accuracy (RandomizedSearchCV):", accuracy_score(y_test, y_pred_rf_random))
print("Classification Report:\n", classification_report(y_test, y_pred_rf_random))


# SVM
from sklearn.svm import SVC

svm = SVC(random_state=42)

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, n_jobs=-1, verbose=2)
grid_svm.fit(X_resampled_pca, y_resampled)

best_svm = grid_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test_pca)
print("Best Parameters (GridSearchCV):", grid_svm.best_params_)
print("SVM Accuracy (GridSearchCV):", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

conf_matrix = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=best_svm.classes_, yticklabels=best_svm.classes_)
plt.title("Confusion Matrix (SVM )")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


param_dist_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1]
}

random_svm = RandomizedSearchCV(estimator=svm, param_distributions=param_dist_svm, n_iter=20, cv=5,
                                n_jobs=-1, verbose=2, random_state=42)
random_svm.fit(X_resampled_pca, y_resampled)

best_svm_random = random_svm.best_estimator_
y_pred_svm_random = best_svm_random.predict(X_test_pca)
print("Best Parameters (RandomizedSearchCV):", random_svm.best_params_)
print("SVM Accuracy (RandomizedSearchCV):", accuracy_score(y_test, y_pred_svm_random))
print("Classification Report:\n", classification_report(y_test, y_pred_svm_random))

# work done by zaimal
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9],
    'subsample': [0.8, 1.0]
}

grid_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=2)
grid_xgb.fit(X_resampled_pca, y_resampled)

best_xgb = grid_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test_pca)
print("Best Parameters (GridSearchCV):", grid_xgb.best_params_)
print("XGBoost Accuracy (GridSearchCV):", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

conf_matrix = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=best_xgb.classes_, yticklabels=best_xgb.classes_)
plt.title("Confusion Matrix ( Xgb)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

param_dist_xgb = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 6, 9],
    'subsample': [0.7, 0.8, 1.0]
}

random_xgb = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist_xgb, n_iter=50, cv=5,
                                n_jobs=-1, verbose=2, random_state=42)
random_xgb.fit(X_resampled_pca, y_resampled)

best_xgb_random = random_xgb.best_estimator_
y_pred_xgb_random = best_xgb_random.predict(X_test_pca)
print("Best Parameters (RandomizedSearchCV):", random_xgb.best_params_)
print("XGBoost Accuracy (RandomizedSearchCV):", accuracy_score(y_test, y_pred_xgb_random))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb_random))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}


grid_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5, n_jobs=-1, verbose=2)
grid_knn.fit(X_resampled_pca, y_resampled)

best_knn = grid_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test_pca)
print("Best Parameters (GridSearchCV):", grid_knn.best_params_)
print("KNN Accuracy (GridSearchCV):", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

conf_matrix = confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=best_knn.classes_, yticklabels=best_knn.classes_)
plt.title("Confusion Matrix ( KNN)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


param_dist_knn = {
    'n_neighbors': [3, 5, 7, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

random_knn = RandomizedSearchCV(estimator=knn, param_distributions=param_dist_knn, n_iter=20, cv=5,
                                n_jobs=-1, verbose=2, random_state=42)
random_knn.fit(X_resampled_pca, y_resampled)

best_knn_random = random_knn.best_estimator_
y_pred_knn_random = best_knn_random.predict(X_test_pca)
print("Best Parameters (RandomizedSearchCV):", random_knn.best_params_)
print("KNN Accuracy (RandomizedSearchCV):", accuracy_score(y_test, y_pred_knn_random))
print("Classification Report:\n", classification_report(y_test, y_pred_knn_random))

# Models acuuracy plot
import matplotlib.pyplot as plt
model_names = ["Random Forest", "SVM", "XGBoost", "KNN"]

accuracies = [
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_svm),
    accuracy_score(y_test, y_pred_xgb),
    accuracy_score(y_test, y_pred_knn)   
]

# Plotting the accuracies
plt.figure(figsize=(8, 6))
plt.bar(model_names, accuracies, color='lightblue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison Across Models')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
