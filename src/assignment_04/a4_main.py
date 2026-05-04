from pathlib import Path
import math
from collections import Counter

# Settings
BASE_DIR = Path(__file__).resolve().parent

# Reusable data loading function
def load_data(file_path, class_index=0):
    X = [] # data features
    y = [] # labels

    path = BASE_DIR / file_path

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            row = line.split(",")

            label = row[class_index]
            features = row[:class_index] + row[class_index + 1:]

            X.append(features)
            y.append(label)

    return X, y

# I(U) = - sum_i (f_i * ln(f_i))
# f_i = probability of class i
def entropy(y):
    counts = Counter(y)
    total = len(y)

    ent = 0.0
    for count in counts.values():
        p = count / total # f_i
        ent -= p * math.log(p)

    return ent

# Get breast + cancer data
X, y = load_data("breast-cancer/breast-cancer.data", class_index=0)

# Example print
print(len(X), len(y))
print("First X:", X[0])
print("First y:", y[0])

# Tests
print("\nEntropy tests:")

print("All same:", entropy(["a", "a", "a"]))     # 0
print("Half-half:", entropy(["a", "b"]))         # 1
print("Real data:", entropy(y))                  # ~0. ...