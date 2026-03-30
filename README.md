# 103A-INxxx-ISP-WSI
### Introduction to Artificial Intelligence @ Warsaw University of Technology

Repository containing assignments and projects developed for the **Introduction to Artificial Intelligence (WSI)** course at the Faculty of Electronics and Information Technology (EiTI).

---

## 📂 Project Structure
```
.
├── src/
│   ├── assignment_01/
│   │   ├── backpack-problem/
│   │   └── steepest-descent/
│   │
│   └── assignment_02/
│
├── venv/            # Local virtual environment (ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

### 🔍 Assignments Description

`assignment_01/backpack-problem/`    — Knapsack problem (Exhaustive & Heuristic)
`assignment_01/steepest-descent/`    — Local search with GUI & CEC2017 support
<br><br>
`assignment_02/`           — (Upcoming) Next set of AI tasks

---

## 🛠 Setup & Installation

### 1. Environment Setup

The project is built using Python 3.12. It is highly recommended to use the provided virtual environment structure:
```
python3 -m venv venv
source venv/bin/activate
```

### 2. Installing Dependencies
#### 2.1. For all systems

Install all required libraries (NumPy, Matplotlib, etc.) using the following command:
```
pip install -r requirements.txt
```
#### 2.2. macOS Specific (Tkinter)
If you are running this on macOS and the GUI does not launch, you must install the python-tk package via Homebrew:
```
brew install python-tk@3.12
```
**4. External Libraries (CEC2017)**
Project uses external library CEC2017, to install it you need to go into the steepest_descent folder and clone the library from git
```
cd src
git clone https://github.com/lukaa12/cec2017-py.git
cp -R cec2017-py/cec2017 .
```
Note: The library is excluded from version control via .gitignore to keep the repository lightweight.

## 🚀 Running assignments
### 1. Using terminal
In terminal enter `WSI` folder, then run:
```
python3 -m src.assignment_XX.[rest_of_the_path_to_starting_file]
```
### 2. Using PyCharm IDE
**Do once:**
* Mark `src` as `Sources Root`:
  * right-click the `src` -> `Mark Directory as` -> `Sources Root`.

**For each assignment do the following steps:**
* Configure Run Profile:
  * Change `Script` to `Module` in the dropdown menu next to the source path field
  * Use the dotted path (e.g., src.assignment_01.steepest_descent.a2_gui).
* `PYTHONPATH`: Ensure following options are enabled:
  * `Add content roots to PYTHONPATH`
  * `Add source roots to PYTHONPATH`

---

