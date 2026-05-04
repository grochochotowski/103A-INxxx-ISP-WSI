from pathlib import Path
import math
from collections import Counter

# =========================== SETUP ===========================
BASE_DIR = Path(__file__).resolve().parent
def load_data(file_path, class_index=0):
    """
        Loads a nominal dataset from a .data file.
        The class column is removed from X and stored in y.
    """
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

# =========================== ID3 HELPERS ===========================
def entropy(y):
    """
        Computes entropy I(U) for a set of class labels y.
        f_i is a probability of class i
        >> I(U) = - sum_i (f_i * ln(f_i))
    """

    counts = Counter(y)
    total = len(y)

    ent = 0.0
    for count in counts.values():
        p = count / total # f_i
        ent -= p * math.log(p)

    return ent

def split_entropy(X, y, attribute_index):
    """
        Computes Inf(d, U), the entropy after splitting U by attribute d.
        >> Inf(d,U) = sum_j (|U_j| / |U|) * I(U_j)
    """

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

def information_gain(X, y, attribute_index):
    """
        Computes information gain for attribute d.
        >> InfGain(d, U) = I(U) - Inf(d,U)
    """
    return entropy(y) - split_entropy(X, y, attribute_index)

def majority_class(y):
    """
        Returns the most frequent class in U.
        Used when there are no attributes left or as a fallback during prediction.
    """
    counts = Counter(y)
    return counts.most_common(1)[0][0]

# =========================== TREE ===========================
class DecisionTreeNode:
    """
    Represents a single node in the ID3 decision tree.

    If label is not None, the node is a leaf.
    Otherwise, the node tests attribute_index and follows children[value].
    majority_label is used as a fallback during prediction.
    """
    def __init__(self, attribute_index=None, label=None, majority_label=None):
        self.attribute_index = attribute_index # attribute used for splitting
        self.label = label # class label if leaf
        self.majority_label = majority_label # most common class
        self.children = {}

# =========================== TESTS ===========================
# Get breast cancer data
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

print("\nMajority class test:")
print("Majority:", majority_class(y))