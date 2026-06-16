from pathlib import Path
import random
import sys


# =========================== IMPORT ID3 FROM ASSIGNMENT 04 ===========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.assignment_04.a4_main import (
    build_tree,              # assignment 4
    predict,                 # assignment 4
    train_test_split,        # assignment 4
    accuracy,                # assignment 4
    confusion_matrix,        # assignment 4
    print_confusion_matrix   # assignment 4
)


# =========================== SETUP ===========================
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

NETWORK_FILE = BASE_DIR / "network.txt"
OUTPUT_FILE = RESULTS_DIR / "examples.txt"


# =========================== LOAD BAYESIAN NETWORK ===========================
def load_network(filename):
    data = {}

    with open(filename, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    data["Chair"] = float(lines[0].split()[1])
    data["Sport"] = float(lines[1].split()[1])

    data["Back"] = {}
    data["Ache"] = {}

    i = 3

    while lines[i] != "Ache":
        chair, sport, probability = lines[i].split()
        data["Back"][(int(chair), int(sport))] = float(probability)
        i += 1

    i += 1

    while i < len(lines):
        back, probability = lines[i].split()
        data["Ache"][int(back)] = float(probability)
        i += 1

    return data


# =========================== GENERATOR ===========================
def generate_example(data):
    chair = 1 if random.random() < data["Chair"] else 0
    sport = 1 if random.random() < data["Sport"] else 0

    p_back = data["Back"][(chair, sport)]
    back = 1 if random.random() < p_back else 0

    p_ache = data["Ache"][back]
    ache = 1 if random.random() < p_ache else 0

    return [chair, sport, back, ache]


def generate_dataset(data, n):
    examples = []

    for _ in range(n):
        examples.append(generate_example(data))

    return examples


# =========================== SAVE DATASET ===========================
def save_dataset(examples, filename):
    with open(filename, "w", encoding="utf-8") as file:
        file.write("Chair Sport Back Ache\n")

        for row in examples:
            file.write(f"{row[0]} {row[1]} {row[2]} {row[3]}\n")


# =========================== PREPARE DATA FOR ID3 ===========================
def prepare_data_for_id3(examples):
    X = []
    y = []

    for row in examples:
        chair = str(row[0])
        sport = str(row[1])
        back = str(row[2])
        ache = str(row[3])

        X.append([chair, sport, back])
        y.append(ache)

    return X, y


# =========================== CLASSIFICATION ===========================
def run_classification(examples):
    X, y = prepare_data_for_id3(examples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.4)  # assignment 4

    attributes = list(range(len(X[0])))

    tree = build_tree(X_train, y_train, attributes)  # assignment 4 - training ID3

    y_pred = predict(tree, X_test)  # assignment 4 - testing ID3

    acc = accuracy(y_test, y_pred)  # assignment 4
    cm = confusion_matrix(y_test, y_pred)  # assignment 4

    print("\n===== WYNIKI KLASYFIKACJI ID3 =====")
    print("Accuracy:", acc)
    print_confusion_matrix(cm)  # assignment 4


# =========================== RUN PROGRAM ===========================
data = load_network(NETWORK_FILE)

n = int(input("Ile przykładów wygenerować? "))

examples = generate_dataset(data, n)

save_dataset(examples, OUTPUT_FILE)

run_classification(examples)

print(f"\nZapisano wygenerowane dane do pliku: {OUTPUT_FILE}")