from pathlib import Path
import math
import random
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

def build_tree(X, y, attributes):
    """
    Builds an ID3 decision tree recursively.

    1. If all examples have the same class -> return leaf.
    2. If there are no attributes left -> return leaf with majority class.
    3. Otherwise choose attribute with maximum information gain.
    """

    # Case 1: all labels are the same
    if len(set(y)) == 1:
        return DecisionTreeNode(label=y[0], majority_label=y[0])

    # Case 2: no attributes left
    if not attributes:
        majority = majority_class(y)
        return DecisionTreeNode(label=majority, majority_label=majority)

    # Choose best attribute
    best_attribute = max(
        attributes,
        key=lambda attribute_index: information_gain(X, y, attribute_index)
    )

    node = DecisionTreeNode(
        attribute_index=best_attribute,
        majority_label=majority_class(y)
    )

    # Create branches for each value of the best attribute
    values = set(sample[best_attribute] for sample in X)

    for value in values:
        X_subset = []
        y_subset = []

        for sample, label in zip(X, y):
            if sample[best_attribute] == value:
                X_subset.append(sample)
                y_subset.append(label)

        remaining_attributes = [
            attribute for attribute in attributes
            if attribute != best_attribute
        ]

        node.children[value] = build_tree(
            X_subset,
            y_subset,
            remaining_attributes
        )

    return node

# =========================== PREDICTION ===========================
def predict_one(node, sample):
    """
    Predicts class label for a single sample using the trained ID3 tree.
    """

    # If node is a leaf, return its class label
    if node.label is not None:
        return node.label

    value = sample[node.attribute_index]

    # If test sample has an unknown attribute value, use fallback class
    if value not in node.children:
        return node.majority_label

    return predict_one(node.children[value], sample)

def predict(tree, X):
    """
    Predicts class labels for all samples in X.
    """
    predictions = []

    for sample in X:
        predictions.append(predict_one(tree, sample))

    return predictions

# =========================== DATA SPLIT ===========================
def train_test_split(X, y, test_ratio=0.4):
    """
    Splits data into train and test sets.

    test_ratio = 0.4 -> 60% train, 40% test (3:2 as required)
    """

    data = list(zip(X, y))
    random.shuffle(data)

    split_index = int(len(data) * (1 - test_ratio))

    train_data = data[:split_index]
    test_data = data[split_index:]

    X_train = [x for x, _ in train_data]
    y_train = [y for _, y in train_data]

    X_test = [x for x, _ in test_data]
    y_test = [y for _, y in test_data]

    return X_train, X_test, y_train, y_test

# =========================== EVALUATION ===========================
def accuracy(y_true, y_pred):
    """
    Computes classification accuracy = number of correct predictions / total number of predictions
    """

    correct = 0

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label == predicted_label:
            correct += 1

    return correct / len(y_true)

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

attributes = list(range(len(X[0])))
tree = build_tree(X, y, attributes)

print("\nPrediction test:")
print("True label:", y[0])
print("Predicted:", predict_one(tree, X[0]))

print("\nTrain/Test split:")

X_train, X_test, y_train, y_test = train_test_split(X, y)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

attributes = list(range(len(X[0])))
tree = build_tree(X_train, y_train, attributes)

y_pred = predict(tree, X_test)

print("\nPrediction on test set:")
print("First true label:", y_test[0])
print("First predicted:", y_pred[0])

print("\nEvaluation:")
print("Accuracy:", accuracy(y_test, y_pred))