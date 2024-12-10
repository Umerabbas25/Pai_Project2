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