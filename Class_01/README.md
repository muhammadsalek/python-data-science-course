#  Python for Data Science, Machine Learning & AI

## Class 01 — Python for Data Science: Complete Foundations

<p align="center">
  <b>Python Basics • NumPy • Pandas • Matplotlib • Google Colab</b><br>
  <i>Beginner-friendly notes, examples, practice tasks, and a first data-science workflow</i>
</p>

---

![Level](https://img.shields.io/badge/Level-Beginner-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.x-yellow?style=for-the-badge&logo=python)
![Colab](https://img.shields.io/badge/Google-Colab-orange?style=for-the-badge&logo=googlecolab)
![NumPy](https://img.shields.io/badge/NumPy-Basics-4D77CF?style=for-the-badge&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Basics-150458?style=for-the-badge&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green?style=for-the-badge)

> **Salek Data Lab**  
> Data • Skills • Research • Impact

---

##  Class 01 at a Glance

```text
PYTHON FOR DATA SCIENCE

Python Basics
    ↓
Variables & Data Types
    ↓
Lists • Dictionaries • Tuples • Sets
    ↓
Conditions • Loops • Functions
    ↓
NumPy Arrays & Numerical Computing
    ↓
Random Data & Basic Statistics
    ↓
Matplotlib Visualization
    ↓
Pandas DataFrame
    ↓
Explore Data
    ↓
Missing Values & Duplicates
    ↓
Ready for Data Cleaning, EDA & Machine Learning
```

---

##  Open the Class Notebook

**Google Colab:**  
https://colab.research.google.com/drive/1V1xVFt3-WVzqBL_hPNKFrXT5k7YVfURX

> Google Colab lets you write and execute Python in a browser without a local Python installation. It is especially useful for students, researchers, and data-science learners.

---

#  Table of Contents

<details>
<summary><strong>Click to expand</strong></summary>

1. [Learning Objectives](#-learning-objectives)
2. [Why Python for Data Science?](#1-why-python-for-data-science)
3. [Google Colab Basics](#2-google-colab-basics)
4. [Variables](#3-variables)
5. [Core Data Types](#4-core-data-types)
6. [Python Collections](#5-python-collections)
7. [Operators](#6-operators)
8. [Conditional Statements](#7-conditional-statements)
9. [Loops](#8-loops)
10. [Functions](#9-functions)
11. [NumPy Basics](#10-numpy-basics)
12. [Arrays, Matrices & Vectorized Operations](#11-arrays-matrices--vectorized-operations)
13. [Random Data & Reproducibility](#12-random-data--reproducibility)
14. [Basic Statistics with NumPy](#13-basic-statistics-with-numpy)
15. [Matplotlib & Histograms](#14-matplotlib--histograms)
16. [Pandas Basics](#15-pandas-basics)
17. [Series vs DataFrame](#16-series-vs-dataframe)
18. [Creating a Practice Dataset](#17-creating-a-practice-dataset)
19. [Exploring a DataFrame](#18-exploring-a-dataframe)
20. [Missing Values](#19-missing-values)
21. [Duplicate Rows](#20-duplicate-rows)
22. [First Data-Science Workflow](#21-first-data-science-workflow)
23. [Common Beginner Mistakes](#22-common-beginner-mistakes)
24. [Mini Project](#23-mini-project)
25. [Practice Exercises](#24-practice-exercises)
26. [Quick Cheat Sheet](#25-quick-cheat-sheet)
27. [Key Terms](#26-key-terms)
28. [Recommended Learning Resources](#27-recommended-learning-resources)
29. [Class Summary](#28-class-summary)

</details>

---

#  Learning Objectives

By the end of Class 01, you should be able to:

- understand how Python is used in data science;
- run Python code in Google Colab;
- create variables and recognize common Python data types;
- work with lists, dictionaries, tuples, and sets;
- write simple conditions, loops, and functions;
- create and manipulate NumPy arrays;
- calculate simple numerical summaries;
- generate reproducible random data;
- create a histogram with Matplotlib;
- create and inspect a pandas DataFrame;
- recognize missing values and duplicate observations;
- understand the basic flow of a data-science project.

---

# 1. Why Python for Data Science?

Python is a general-purpose programming language widely used for:

- data analysis;
- statistics;
- scientific computing;
- machine learning;
- deep learning;
- visualization;
- automation;
- web applications;
- research workflows.

For data science, Python becomes especially powerful through libraries such as:

| Library | Main Purpose |
|---|---|
| **NumPy** | Numerical computing and arrays |
| **pandas** | Tabular data manipulation |
| **Matplotlib** | Data visualization |
| **scikit-learn** | Machine learning |
| **SciPy** | Scientific computing |
| **TensorFlow / PyTorch** | Deep learning |

> **Simple idea:** Python is the language; libraries give Python specialized data-science abilities.

---

# 2. Google Colab Basics

Google Colab is a hosted notebook environment.

### Why beginners like Colab

- no installation is required;
- Python runs in the browser;
- notebooks contain both code and notes;
- work can be shared easily;
- many popular data-science libraries are already available.

### Two important cell types

| Cell | Purpose |
|---|---|
| **Code cell** | Run Python code |
| **Text / Markdown cell** | Write explanations, formulas, headings, and notes |

### Your first code

```python
print("Hello, Data Science!")
```

### Useful shortcuts

```text
Ctrl + Enter   → Run current cell
Shift + Enter  → Run cell and move to the next cell
```

---

# 3. Variables

A **variable** is a name that refers to a value stored in memory.

```python
name = "Salek Data Lab"
year = 2026
pi = 3.1416
is_ready = True
```

Think of variables as labeled containers:

```text
name      → "Salek Data Lab"
year      → 2026
pi        → 3.1416
is_ready  → True
```

### Display values

```python
print(name)
print(year)
```

### Good variable names

```python
student_age = 24
mean_score = 82.5
sample_size = 500
```

### Avoid unclear names

```python
a = 24
x1 = 82.5
z = 500
```

Meaningful names make analysis easier to read and reproduce.

---

# 4. Core Data Types

Python automatically identifies the type of a value.

```python
name = "Salek Data Lab"
year = 2026
pi = 3.1416
is_ready = True

print(type(name))
print(type(year))
print(type(pi))
print(type(is_ready))
```

| Type | Meaning | Example |
|---|---|---|
| `str` | Text | `"Python"` |
| `int` | Whole number | `2026` |
| `float` | Decimal number | `3.1416` |
| `bool` | Logical value | `True`, `False` |
| `NoneType` | Missing / no value | `None` |

### Type conversion

```python
age_text = "24"
age = int(age_text)

score = 95
score_decimal = float(score)

year = 2026
year_text = str(year)
```

---

# 5. Python Collections

Python provides several structures for storing multiple values.

## 5.1 List

A **list** is ordered and mutable.

```python
scores = [100, 50, 200, 150]
print(scores)
```

> Lists use **square brackets `[]`**.

### Indexing

Python starts counting at `0`.

```python
print(scores[0])
print(scores[1])
```

### Slicing

```python
print(scores[:2])
print(scores[1:3])
```

### Common list operations

```python
scores.append(180)
print(len(scores))
print(sum(scores))
print(min(scores))
print(max(scores))
```

### Average

```python
average = sum(scores) / len(scores)
print("Average:", average)
```

---

## 5.2 Dictionary

A **dictionary** stores information as `key: value` pairs.

```python
student = {
    "name": "Salek",
    "age": 24,
    "department": "Statistics",
    "score": 100
}
```

> Dictionaries use **curly braces `{}`**.

Access a value:

```python
print(student["name"])
print(student["department"])
```

Add or update a value:

```python
student["score"] = 95
student["university"] = "SUST"
```

---

## 5.3 Tuple

A **tuple** is ordered but immutable.

```python
coordinates = (20.5, 50.5)
print(coordinates)
```

> Tuples commonly use **parentheses `()`**.

Use a tuple when values should not normally change.

---

## 5.4 Set

A **set** stores unique values.

```python
scores = [100, 50, 100, 150, 50]
unique_scores = set(scores)

print(unique_scores)
```

A set is useful when you want to remove repeated values or perform set operations.

---

## 5.5 Quick Comparison

| Structure | Ordered | Mutable | Duplicates | Typical Use |
|---|---:|---:|---:|---|
| List | Yes | Yes | Yes | Collection of observations |
| Tuple | Yes | No | Yes | Fixed collection |
| Dictionary | Yes* | Yes | Keys unique | Key-value information |
| Set | No fixed positional indexing | Yes | No | Unique values |

\*Modern Python dictionaries preserve insertion order.

---

# 6. Operators

Operators allow Python to perform calculations and comparisons.

## Arithmetic operators

```python
a = 10
b = 3

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** b)
print(a % b)
```

| Operator | Meaning |
|---|---|
| `+` | Addition |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division |
| `**` | Power |
| `%` | Remainder |

## Comparison operators

```python
score = 85

print(score > 80)
print(score >= 80)
print(score == 85)
print(score != 70)
```

## Logical operators

```python
age = 25
smoker = False

print(age >= 18 and smoker == False)
```

Main logical operators:

```text
and
or
not
```

---

# 7. Conditional Statements

Conditions let a program make decisions.

```python
score = 85

if score >= 80:
    grade = "A"
elif score >= 70:
    grade = "B"
else:
    grade = "C"

print(grade)
```

### Flow

```text
Is score >= 80?
      │
   Yes│No
      │
      A
      ↓
Is score >= 70?
   │       │
 Yes      No
   │       │
   B       C
```

### Important: indentation

Python uses indentation to define blocks of code.

Correct:

```python
if score >= 80:
    print("High score")
```

Incorrect:

```python
if score >= 80:
print("High score")
```

---

# 8. Loops

Loops repeat actions.

## `for` loop

```python
scores = [100, 50, 200, 150]

for score in scores:
    print(score)
```

## Condition inside a loop

```python
for score in scores:
    if score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    else:
        grade = "C"

    print(score, grade)
```

## `range()`

```python
for i in range(5):
    print(i)
```

Output:

```text
0
1
2
3
4
```

## `while` loop

```python
count = 1

while count <= 5:
    print(count)
    count += 1
```

---

# 9. Functions

A **function** is a reusable block of code.

```python
def calculate_bmi(weight, height):
    return weight / (height ** 2)

bmi = calculate_bmi(72, 1.65)
print("BMI:", round(bmi, 2))
```

### Function anatomy

```text
def calculate_bmi(weight, height):
│       │             │
│       │             └── parameters
│       └── function name
└── defines a function
```

### `return`

`return` sends a result back to the place where the function was called.

```python
def square(x):
    return x ** 2

result = square(5)
print(result)
```

> **Important:** Code written after `return` inside the same function block will not run. Call the function outside the function definition, as shown above.

---

# 10. NumPy Basics

NumPy stands for **Numerical Python**.

It provides fast, efficient multidimensional arrays and numerical operations.

### Import NumPy

```python
import numpy as np
```

`np` is the conventional alias.

### Create an array

```python
x = np.array([1, 2, 3, 4, 5])
print(x)
```

### Why use NumPy instead of only Python lists?

NumPy is designed for numerical computing and supports:

- multidimensional arrays;
- fast vectorized operations;
- mathematical functions;
- matrix operations;
- random-number generation;
- integration with pandas, Matplotlib, SciPy, and machine-learning libraries.

---

# 11. Arrays, Matrices & Vectorized Operations

## 1-D array

```python
x = np.array([1, 2, 3, 4, 5])
```

## Element-wise operation

```python
print(x ** 2)
```

Output:

```text
[ 1  4  9 16 25]
```

No manual loop is needed.

## 2-D array / matrix

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix)
```

### Useful array attributes

```python
print(matrix.shape)
print(matrix.ndim)
print(matrix.size)
print(matrix.dtype)
```

| Attribute | Meaning |
|---|---|
| `.shape` | Dimensions |
| `.ndim` | Number of dimensions |
| `.size` | Total elements |
| `.dtype` | Data type |

---

# 12. Random Data & Reproducibility

Random data are useful for:

- simulation;
- teaching;
- testing code;
- probability experiments;
- machine-learning demonstrations.

## Traditional seed example

```python
np.random.seed(42)

random_data = np.random.normal(
    loc=100,
    scale=15,
    size=1000
)
```

### Parameters

| Parameter | Meaning |
|---|---|
| `loc=100` | Mean |
| `scale=15` | Standard deviation |
| `size=1000` | Number of observations |

### Why use a seed?

A random seed helps reproduce the same pseudo-random sequence.

```text
Same seed → same reproducible example
```

## Modern NumPy random generator

```python
rng = np.random.default_rng(42)

random_data = rng.normal(
    loc=100,
    scale=15,
    size=1000
)
```

For new code, `default_rng()` is a clean modern approach.

---

# 13. Basic Statistics with NumPy

```python
x = np.array([1, 2, 3, 4, 5])

print("Mean:", np.mean(x))
print("Median:", np.median(x))
print("SD:", np.std(x, ddof=1))
print("Minimum:", np.min(x))
print("Maximum:", np.max(x))
```

### Sample vs population standard deviation

```python
np.std(x, ddof=0)   # population-style denominator N
np.std(x, ddof=1)   # sample-style denominator N - 1
```

For many research summaries of a sample, `ddof=1` is commonly used.

---

# 14. Matplotlib & Histograms

Matplotlib is a major Python visualization library.

```python
import matplotlib.pyplot as plt
```

## Histogram

```python
plt.figure(figsize=(8, 5))
plt.hist(random_data, bins=30, edgecolor="black")
plt.title("Normal Random Sample")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

### What does a histogram show?

A histogram helps us inspect the distribution of numerical data.

It can reveal:

- center;
- spread;
- skewness;
- multiple peaks;
- unusual values;
- approximate distribution shape.

### What are bins?

Bins divide a numeric range into intervals.

```text
Values → intervals (bins) → counts → bars
```

Changing the number of bins can change how the distribution appears, so bin choice should be thoughtful.

---

# 15. Pandas Basics

pandas is one of the most important libraries for tabular data analysis.

```python
import pandas as pd
```

Typical tasks include:

- importing data;
- selecting rows and columns;
- cleaning values;
- managing missing data;
- grouping observations;
- creating variables;
- merging datasets;
- calculating summaries;
- preparing data for analysis and machine learning.

---

# 16. Series vs DataFrame

## Series

A **Series** is a one-dimensional labeled array.

```python
ages = pd.Series([20, 21, 23, 25])
print(ages)
```

## DataFrame

A **DataFrame** is a two-dimensional labeled table.

```python
df = pd.DataFrame({
    "name": ["A", "B", "C"],
    "age": [20, 21, 23]
})

print(df)
```

Think of a DataFrame as similar to:

- an Excel worksheet;
- an SQL table;
- a data frame in R.

---

# 17. Creating a Practice Dataset

The Class 01 notebook creates a synthetic dataset so that we can practice without downloading external data.

```python
rng = np.random.default_rng(42)
n = 500

df = pd.DataFrame({
    "id": np.arange(1, n + 1),
    "age": rng.integers(18, 80, n),
    "gender": rng.choice(
        ["Female", "Male", "female", "male"],
        n
    ),
    "bmi": rng.normal(25, 4, n),
    "smoker": rng.choice(
        ["Yes", "No", None],
        n,
        p=[0.20, 0.75, 0.05]
    ),
    "outcome": rng.binomial(1, 0.25, n)
})
```

### Variables

| Variable | Example Type | Meaning |
|---|---|---|
| `id` | Integer | Observation identifier |
| `age` | Numeric | Age |
| `gender` | Categorical | Gender label |
| `bmi` | Numeric | Body mass index |
| `smoker` | Categorical | Smoking status |
| `outcome` | Binary | 0/1 outcome |

### Why intentionally include `"Female"` and `"female"`?

Because real datasets often contain inconsistent categories.

Later, during data cleaning, we can standardize them.

```text
"Female"
"female"
"FEMALE"
```

These may represent the same category but appear differently to Python.

---

# 18. Exploring a DataFrame

Before analysis, always inspect the dataset.

## First rows

```python
df.head()
```

## Dimensions

```python
df.shape
```

Returns:

```text
(number_of_rows, number_of_columns)
```

## Data types

```python
df.dtypes
```

## Summary statistics

```python
df.describe()
```

For additional information:

```python
df.info()
```

Check column names:

```python
df.columns
```

Count missing values:

```python
df.isna().sum()
```

---

# 19. Missing Values

Missing values are common in real-world datasets.

In NumPy:

```python
np.nan
```

In Python object/categorical contexts, missingness may also appear as:

```python
None
```

## Add missing BMI values for practice

```python
missing_rows = rng.choice(
    df.index,
    25,
    replace=False
)

df.loc[missing_rows, "bmi"] = np.nan
```

## Detect missing values

```python
df.isna().sum()
```

## Missing percentage

```python
df.isna().mean() * 100
```

> Do not automatically delete missing observations. The correct strategy depends on why the values are missing, how much is missing, the variable type, and the analysis goal.

---

# 20. Duplicate Rows

Duplicate observations can distort analysis.

The class notebook intentionally adds duplicate rows:

```python
df = pd.concat(
    [df, df.iloc[:5]],
    ignore_index=True
)
```

## Detect duplicates

```python
df.duplicated().sum()
```

## View duplicates

```python
df[df.duplicated()]
```

## Remove exact duplicates

```python
df_clean = df.drop_duplicates()
```

> In real research data, confirm that rows are truly duplicates before removing them.

---

# 21. First Data-Science Workflow

```text
1. Define the problem
       ↓
2. Obtain data
       ↓
3. Inspect the data
       ↓
4. Understand variables and data types
       ↓
5. Check missing values
       ↓
6. Check duplicates and inconsistencies
       ↓
7. Clean / transform data
       ↓
8. Explore and visualize
       ↓
9. Statistical analysis / machine learning
       ↓
10. Evaluate and interpret
       ↓
11. Communicate results
       ↓
12. Reproduce and share the workflow
```

Class 01 introduces the first building blocks of this workflow.

---

# 22. Common Beginner Mistakes

## Mistake 1 — Confusing Python brackets

```text
List        → [ ]
Dictionary  → { key: value }
Tuple       → ( )
Set         → { } or set(...)
```

## Mistake 2 — Forgetting indentation

```python
if age >= 18:
    print("Adult")
```

## Mistake 3 — Using `=` instead of `==`

```python
age = 24       # assignment
age == 24      # comparison
```

## Mistake 4 — Calling a function after `return` but still inside the function

Incorrect:

```python
def calculate_bmi(weight, height):
    return weight / (height ** 2)
    bmi = calculate_bmi(72, 1.65)
```

The line after `return` cannot execute.

Correct:

```python
def calculate_bmi(weight, height):
    return weight / (height ** 2)

bmi = calculate_bmi(72, 1.65)
print(round(bmi, 2))
```

## Mistake 5 — Ignoring capitalization in categories

```text
Male
male
MALE
```

Python treats these as different strings.

## Mistake 6 — Assuming missing values are automatically handled

Always inspect:

```python
df.isna().sum()
```

## Mistake 7 — Editing data before inspecting it

Start with:

```python
df.head()
df.shape
df.dtypes
df.info()
df.describe()
```

---

# 23. Mini Project

## 🎓 Student Health Data Explorer

Create the following dataset:

```python
students = pd.DataFrame({
    "name": ["Ayesha", "Rahim", "Nadia", "Karim", "Sara"],
    "age": [22, 23, 21, 24, 22],
    "weight": [52, 70, 58, 76, 60],
    "height": [1.60, 1.75, 1.63, 1.78, 1.66]
})
```

### Task 1 — Create a BMI function

```python
def calculate_bmi(weight, height):
    return weight / (height ** 2)
```

### Task 2 — Calculate BMI

Try to create a new `bmi` column.

### Task 3 — Explore the data

Find:

- number of rows and columns;
- mean age;
- mean BMI;
- minimum and maximum weight.

### Task 4 — Visualize BMI

Create a histogram of BMI.

### Task 5 — Explain your result

Write 2–3 sentences describing what you observe.

---

# 24. Practice Exercises

## Beginner

1. Create variables for your name, age, department, and current year.
2. Print the type of each variable.
3. Create a list of five exam scores.
4. Calculate the average score.
5. Create a dictionary describing one student.
6. Convert the score list to a set.
7. Use an `if` statement to classify a score as pass/fail.
8. Use a loop to print each score.
9. Write a function that converts Celsius to Fahrenheit.
10. Create a NumPy array from 1 to 10.

## Data Science Practice

11. Calculate the mean, median, SD, minimum, and maximum of a NumPy array.
12. Generate 1,000 random normal values.
13. Plot a histogram.
14. Create a pandas DataFrame with at least four columns.
15. Display the first five rows.
16. Inspect `shape`, `dtypes`, and `describe()`.
17. Insert some missing values.
18. Count missing observations.
19. Add duplicate rows.
20. Count and remove duplicates.

## Challenge

Create a small synthetic research dataset with:

- `id`;
- `age`;
- `sex`;
- `bmi`;
- `exposure`;
- `outcome`.

Then perform a basic inspection and summarize what you find.

---

# 25. Quick Cheat Sheet

## Python

```python
# Variable
x = 10

# Print
print(x)

# Type
type(x)

# List
a = [1, 2, 3]

# Dictionary
person = {"name": "Salek", "age": 24}

# Tuple
point = (10, 20)

# Set
unique = set([1, 1, 2, 3])

# Condition
if x > 5:
    print("High")

# Loop
for item in a:
    print(item)

# Function
def square(x):
    return x ** 2
```

## NumPy

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])

np.mean(x)
np.median(x)
np.std(x, ddof=1)
x.shape
x ** 2
```

## pandas

```python
import pandas as pd

df.head()
df.shape
df.dtypes
df.info()
df.describe()
df.isna().sum()
df.duplicated().sum()
```

## Matplotlib

```python
import matplotlib.pyplot as plt

plt.hist(x)
plt.show()
```

---

# 26. Key Terms

| Term | Simple Meaning |
|---|---|
| Python | Programming language |
| Variable | Named reference to a value |
| Data type | Kind of value stored |
| List | Ordered mutable collection |
| Dictionary | Key-value collection |
| Tuple | Ordered immutable collection |
| Set | Collection of unique values |
| Condition | Decision rule |
| Loop | Repeats code |
| Function | Reusable block of code |
| Library | Collection of reusable tools |
| NumPy | Numerical array library |
| `ndarray` | NumPy multidimensional array |
| pandas | Tabular data-analysis library |
| Series | 1-D labeled pandas structure |
| DataFrame | 2-D labeled table |
| Matplotlib | Visualization library |
| Missing value | Observation with no recorded value |
| Duplicate | Repeated observation |
| Seed | Starting state for reproducible pseudo-random generation |
| EDA | Exploratory Data Analysis |

---

# 27. Recommended Learning Resources

These resources are useful companions to the course. Start with beginner sections rather than trying to read everything.

##  Python

### Python Official Tutorial
https://docs.python.org/3/tutorial/

Recommended beginner topics:

- basic syntax;
- control flow;
- functions;
- data structures;
- modules.

### Kaggle Learn — Python
https://www.kaggle.com/learn/python

Especially useful for short hands-on lessons on:

- variables;
- functions;
- booleans;
- conditions;
- lists;
- loops;
- dictionaries;
- external libraries.

---

##  NumPy

### NumPy Learn
https://numpy.org/learn/

### NumPy — Absolute Basics for Beginners
https://numpy.org/doc/stable/user/absolute_beginners.html

Focus on:

- importing NumPy;
- arrays;
- shape and dimensions;
- indexing;
- basic array operations.

---

##  pandas

### pandas Getting Started
https://pandas.pydata.org/getting_started.html

### pandas Intro Tutorials
https://pandas.pydata.org/docs/getting_started/intro_tutorials/

Focus on:

- DataFrame;
- selecting data;
- reading/writing tabular data;
- summary statistics;
- missing data;
- combining and reshaping tables.

---

##  Matplotlib

### Matplotlib Tutorials
https://matplotlib.org/stable/tutorials/

Start with:

- Quick Start;
- pyplot basics;
- titles and labels;
- histograms;
- scatter plots;
- line and bar plots.

---

## ☁️ Google Colab

### Welcome to Google Colab
https://colab.research.google.com/notebooks/welcome.ipynb

Practice:

- creating code cells;
- creating Markdown cells;
- running cells;
- restarting runtime;
- uploading files;
- saving notebooks.

---

# 28. Class Summary

```text
CLASS 01 COMPLETE

✓ Python & Google Colab
✓ Variables
✓ Core data types
✓ Lists
✓ Dictionaries
✓ Tuples
✓ Sets
✓ Operators
✓ Conditions
✓ Loops
✓ Functions
✓ NumPy arrays
✓ Matrices
✓ Random data
✓ Basic numerical statistics
✓ Matplotlib histogram
✓ pandas DataFrame
✓ Data exploration
✓ Missing values
✓ Duplicate rows
✓ First data-science workflow
```

### What comes next?

**Class 02 — Python Programming for Data Work**

Topics can include:

- deeper conditional logic;
- loops;
- functions;
- parameters and arguments;
- list/dictionary comprehensions;
- lambda functions;
- modules;
- file handling;
- error handling;
- practical data-science coding patterns.

---

#  Suggested Homework

Create your own small dataset and demonstrate:

1. Python variables and data types;
2. at least two Python collections;
3. one conditional statement;
4. one loop;
5. one function;
6. one NumPy array;
7. one pandas DataFrame;
8. one descriptive summary;
9. one histogram;
10. missing-value and duplicate checks.

---

#  Salek Data Lab

**Python for Data Science, Machine Learning & AI**

> Learn → Code → Analyze → Explain → Build → Deploy

If this material helps you, practice every example yourself rather than only reading the code.

---
