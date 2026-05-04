from pathlib import Path

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


# Get breast + cancer data
X, y = load_data("breast-cancer/breast-cancer.data", class_index=0)

# Example print
print(len(X), len(y))
print("First X:", X[0])
print("First y:", y[0])