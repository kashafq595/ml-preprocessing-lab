import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Dataset load
df = pd.read_csv("dataset/Bank-Customer-Churn-Prediction.csv")

print(df.head())

# Part 1, Task 1: Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
#plt.show()

# Task 2: churn ke saath correlation
corr_target = df.corr(numeric_only=True)['churn'].sort_values()
print("\nCorrelation with churn:")
print(corr_target)

# Task 3: |correlation| > 0.1 wale features
selected = corr_target[(corr_target > 0.1) | (corr_target < -0.1)]
print("\nSelected features (corr > 0.1 or < -0.1):")
print(selected)

# Task 4: customer_id hatao
df = df.drop(columns=['customer_id'])
print("\nColumns after dropping customer_id:")
print(df.columns.tolist())

# ---------- Part 2: Imbalanced Data ----------

# Task 1: Class distribution
print("\nClass distribution:")
print(df['churn'].value_counts())

# Task 2: Percentage mein
print("\nClass distribution (%):")
print(df['churn'].value_counts(normalize=True) * 100)

# Task 3: Upsampling function
def upsample(features, target, repeat):
    features_0 = features[target == 0]
    features_1 = features[target == 1]
    target_0 = target[target == 0]
    target_1 = target[target == 1]
    features_up = pd.concat([features_0] + [features_1] * repeat)
    target_up = pd.concat([target_0] + [target_1] * repeat)
    features_up, target_up = shuffle(features_up, target_up, random_state=42)
    return features_up, target_up

features = df.drop("churn", axis=1)
target = df["churn"]
features_up, target_up = upsample(features, target, 3)

# Task 4: Upsampling ke baad distribution
print("\nClass distribution after upsampling:")
print(target_up.value_counts())
print(target_up.value_counts(normalize=True) * 100)


# Task 5: Downsampling function
def downsample(features, target, fraction):
    features_0 = features[target == 0]
    features_1 = features[target == 1]
    target_0 = target[target == 0]
    target_1 = target[target == 1]
    features_down = pd.concat([features_0.sample(frac=fraction, random_state=42), features_1])
    target_down = pd.concat([target_0.sample(frac=fraction, random_state=42), target_1])
    features_down, target_down = shuffle(features_down, target_down, random_state=42)
    return features_down, target_down

features_down, target_down = downsample(features, target, 0.25)

print("\nClass distribution after downsampling:")
print(target_down.value_counts())
print(target_down.value_counts(normalize=True) * 100)


# ---------- Part 3: Encoding Categorical Variables ----------

# Task 1: Categorical columns pehchano
print("\nCategorical columns:")
print(df.select_dtypes(include=['object', 'string']).head())

# Task 2: One-Hot Encoding (aapke columns: country, gender)
cols_before = df.shape[1]
df = pd.get_dummies(df, columns=['country', 'gender'], drop_first=True)

# Task 3: Dataset dobara print karo
print("\nAfter One-Hot Encoding:")
print(df.head())
print("\nColumns:", df.columns.tolist())
print("Columns before:", cols_before, "| after:", df.shape[1])

# ---------- Part 4: Data Splitting ----------

# Task 1: Train+valid aur test (80% / 20%)
train_valid, test = train_test_split(df, test_size=0.2, random_state=42)

# Task 2: Train or validation (train_valid 75% / 25%)
train, valid = train_test_split(train_valid, test_size=0.25, random_state=42)

# Task 3: Features or target 
X_train = train.drop("churn", axis=1)
y_train = train["churn"]
X_valid = valid.drop("churn", axis=1)
y_valid = valid["churn"]
X_test = test.drop("churn", axis=1)
y_test = test["churn"]

# Task 4: Shapes print
print("\nTraining set  :", X_train.shape, y_train.shape)
print("Validation set:", X_valid.shape, y_valid.shape)
print("Test set      :", X_test.shape, y_test.shape)