import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#1. Load dan encoding data
df = pd.read_csv('dataeksgiboost.csv', sep=';')
df.columns = df.columns.str.strip()
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

#Cek distribusi data untuk lihat imbalance
print("Distribusi data Attrition (0 = No, 1 = Yes):")
print(df['Attrition'].value_counts())

df_encoded = pd.get_dummies(df, drop_first=True)

#2. Split data
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

#Hitung rasio kelas untuk scale_pos_weight
pos = sum(y_train)
neg = len(y_train) - pos
ratio = neg / pos

#3. Train model dengan scale_pos_weight
model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', scale_pos_weight=ratio)
model.fit(X_train, y_train)

#4. Evaluasi hasil
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ✅ Visualisasi fitur penting
plot_importance(model)
plt.tight_layout()
plt.show()
