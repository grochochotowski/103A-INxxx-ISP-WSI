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

# Inf(d,U) = sum_j (|U_j| / |U|) * I(U_j)
def split_entropy(X, y, attribute_index):
    total = len(y)
    subsets = {}

    # attribute grouping
    for sample, label in zip(X, y):
        value = sample[attribute_index]

        if value not in subsets:
            subsets[value] = []

        subsets[value].append(label)

    # weighted entropy sum
    result = 0.0

    for subset in subsets.values():
        weight = len(subset) / total
        result += weight * entropy(subset)

    return result

# InfGain(d, U) = I(U) - Inf(d,U)
def information_gain(X, y, attribute_index):
    return entropy(y) - split_entropy(X, y, attribute_index)

def majority_class(y):
    counts = Counter(y)
    return counts.most_common(1)[0][0]

# Get breast + cancer data
X, y = load_data("breast-cancer/breast-cancer.data", class_index=0)

# Example print
print(len(X), len(y))
print("First X:", X[0])
print("First y:", y[0])

# Tests
print("\nEntropy tests:")

print("All same:", entropy(["a", "a", "a"]))     # 0
print("Half-half:", entropy(["a", "b"]))         # ln(2) ~= 0.693
print("Real data:", entropy(y))                  # ~0. ...

print("\nInformation Gain tests:")

for attribute_index in range(len(X[0])):
    gain = information_gain(X, y, attribute_index)
    print(f"Attribute {attribute_index}: {gain}")

best_attribute = max(
    range(len(X[0])),
    key=lambda attribute_index: information_gain(X, y, attribute_index)
)

print("Best attribute:", best_attribute)

