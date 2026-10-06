# 🐍 Python for Data Science, Machine Learning & AI

## Class 02 — NumPy, Pandas, Data Cleaning, Visualization & EDA

<p align="center">
<b>NumPy • Pandas • Data Cleaning • Matplotlib • Seaborn • Plotly • Descriptive Statistics • EDA • Basic Inferential Statistics</b>
</p>

![Level](https://img.shields.io/badge/Level-Beginner%20→%20Intermediate-blue?style=for-the-badge)
![Duration](https://img.shields.io/badge/Class-4%20Hours-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.x-yellow?style=for-the-badge&logo=python)
![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?style=for-the-badge&logo=googlecolab)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Python-4D77CF?style=for-the-badge&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)

> **Salek Data Lab**  
> Data • Skills • Research • Impact

---

## 🚀 Open in Google Colab

After uploading this notebook to the repository as:

`Class_02/Python_Class_02_NumPy_Pandas_EDA.ipynb`

you can use:

https://colab.research.google.com/github/muhammadsalek/python-data-science-course/blob/main/Class_02/Python_Class_02_NumPy_Pandas_EDA.ipynb

---

## 📂 Dataset

**Titanic Dataset — Kaggle**  
https://www.kaggle.com/datasets/yasserh/titanic-dataset

For the live class, download `Titanic-Dataset.csv` and upload it to Google Colab when the notebook asks for the file.

---

# 🎯 Class Goal

By the end of this 4-hour class, students should be able to move through a complete beginner-friendly data workflow:

```text
RAW DATA
   ↓
IMPORT & AUDIT
   ↓
NUMPY FUNDAMENTALS
   ↓
PANDAS DATA MANAGEMENT
   ↓
CLEAN LABELS / TYPES / MISSINGNESS / DUPLICATES
   ↓
CREATE & TRANSFORM VARIABLES
   ↓
RESHAPE / MERGE / GROUP / AGGREGATE
   ↓
DESCRIPTIVE STATISTICS
   ↓
VISUALIZATION
   ↓
EDA
   ↓
BASIC INFERENCE
   ↓
EXPORT CLEAN DATA + FIGURES
```

---

# ⏱️ Suggested 4-Hour Teaching Plan

| Time | Module | Main Topics |
|---|---|---|
| 0:00–0:15 | Setup & Dataset Audit | Colab, imports, load Titanic, shape, columns, types, missingness |
| 0:15–1:00 | NumPy | Arrays, dimensions, indexing, slicing, vectorization, broadcasting, statistics, linear algebra, random data |
| 1:00–2:00 | Pandas | Series, DataFrame, import/export, selection, filtering, sorting, ranking, variables, groupby |
| 2:00–2:45 | Cleaning & Wrangling | labels, types, impossible values, missing values, duplicates, outliers, map/apply, merge/join/concat, reshape, pivot |
| 2:45–3:30 | Visualization | Matplotlib, Seaborn, Plotly, histogram, bar, box, scatter, line, count, heatmap, pair plot, subplots |
| 3:30–3:55 | EDA & Statistics | descriptive summaries, distributions, correlation, cross-tabs, uni/bi/multivariate EDA |
| 3:55–4:00+ | Inference & Export | CI, t-test, chi-square, save figures and cleaned dataset |

---

# 📚 Table of Contents

<details>
<summary><strong>Click to expand</strong></summary>

1. NumPy for Scientific Computing
2. Arrays and Dimensions
3. Indexing and Slicing
4. Array Operations
5. Broadcasting
6. Vectorization
7. Mathematical & Statistical Functions
8. Linear Algebra
9. Random Number Generation
10. pandas Series and DataFrame
11. Importing CSV / Excel / JSON / Stata
12. Dataset Audit
13. Selecting & Filtering
14. Sorting & Ranking
15. Creating & Modifying Variables
16. Data Types & Categorical Variables
17. Missing Data
18. Duplicate Detection
19. Impossible Values
20. Outlier Detection
21. `map()` / `apply()`
22. `groupby()` & Aggregation
23. Merge / Join / Concatenate
24. Reshaping
25. Pivot Tables
26. Exporting Cleaned Data
27. Descriptive Statistics
28. Matplotlib
29. Seaborn
30. Plotly
31. Publication-Quality Figures
32. Univariate EDA
33. Bivariate EDA
34. Multivariate EDA
35. Correlation Analysis
36. Cross-Tabulation
37. Automated EDA
38. Research-Oriented EDA Workflow
39. Confidence Interval
40. Independent-Samples t-test
41. Chi-Square Test
42. Class Summary & Exercises

</details>

---

# 1. NumPy for Scientific Computing

**NumPy** provides fast numerical arrays and mathematical operations.

```python
import numpy as np
```

Main ideas:

- an array is a structured collection of values;
- NumPy arrays can have one or more dimensions;
- indexing selects individual elements;
- slicing selects ranges;
- vectorization performs operations on many values at once;
- broadcasting allows compatible arrays of different shapes to interact;
- NumPy includes statistics, random generation and linear algebra.

### Core Array Properties

```python
x = np.array([10, 20, 30, 40, 50])

x.shape
x.ndim
x.size
x.dtype
```

| Property | Meaning |
|---|---|
| `shape` | dimensions of the array |
| `ndim` | number of dimensions |
| `size` | total number of elements |
| `dtype` | element data type |

### Indexing and Slicing

```python
x[0]       # first value
x[-1]      # last value
x[1:4]     # positions 1, 2, 3
x[:3]      # first three values
x[::2]     # every second value
```

### Vectorization

```python
x * 2
x ** 2
np.sqrt(x)
```

Instead of manually looping through every value, NumPy applies the operation to the whole array.

### Broadcasting

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

matrix + 10
```

NumPy automatically applies the scalar `10` to every element.

### Statistics

```python
np.mean(x)
np.median(x)
np.var(x, ddof=1)
np.std(x, ddof=1)
np.percentile(x, [25, 50, 75])
```

### Linear Algebra

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A @ B
np.linalg.det(A)
np.linalg.inv(A)
```

---

# 2. pandas Fundamentals

pandas is designed for labeled tabular data.

```python
import pandas as pd
```

Two core objects:

| Object | Description |
|---|---|
| `Series` | one-dimensional labeled data |
| `DataFrame` | two-dimensional labeled table |

### Series

```python
ages = pd.Series([22, 24, 21, 28])
```

### DataFrame

```python
students = pd.DataFrame({
    "name": ["A", "B", "C"],
    "age": [20, 21, 23]
})
```

---

# 3. Importing Data

```python
pd.read_csv("file.csv")
pd.read_excel("file.xlsx")
pd.read_json("file.json")
pd.read_stata("file.dta")
```

The notebook includes a safe demonstration of all four formats.

---

# 4. Audit Before Cleaning

Never begin analysis by immediately changing the data.

Start with:

```python
df.head()
df.tail()
df.sample()
df.shape
df.columns
df.dtypes
df.info()
df.describe()
df.isna().sum()
df.duplicated().sum()
```

A good audit asks:

1. How many rows and columns?
2. What does one row represent?
3. Which variables are numeric?
4. Which are categorical?
5. Which variables have missing values?
6. Are any labels inconsistent?
7. Are there impossible values?
8. Are there duplicated rows?

---

# 5. Selecting and Filtering

### Select columns

```python
df["Age"]
df[["Age", "Sex", "Survived"]]
```

### `loc`

```python
df.loc[0:5, ["Age", "Sex"]]
```

### `iloc`

```python
df.iloc[0:5, 0:4]
```

### Filter rows

```python
df[df["Age"] >= 18]
df[(df["Sex"] == "female") & (df["Pclass"] == 1)]
```

### Query syntax

```python
df.query("Age >= 18 and Pclass == 1")
```

---

# 6. Sorting and Ranking

```python
df.sort_values("Fare", ascending=False)
df["Fare"].rank(method="average", ascending=False)
```

---

# 7. Data Cleaning & Transformation

A research-oriented cleaning workflow:

```text
Original Dataset
      ↓
Standardize column names
      ↓
Clean category labels
      ↓
Check/convert data types
      ↓
Check impossible values
      ↓
Check missingness
      ↓
Check duplicates
      ↓
Flag outliers
      ↓
Create derived variables
      ↓
Re-audit
      ↓
Save clean dataset
```

### Missing Values

```python
df.isna().sum()
df.isna().mean() * 100
```

Do not automatically delete missing records.

For the Titanic teaching example, the notebook demonstrates sensible simple strategies such as:

- group-median imputation for `Age`;
- mode imputation for `Embarked`;
- creating an indicator for whether `Cabin` was recorded;
- replacing missing cabin text with `"Unknown"` for descriptive work.

> In real research, the missing-data strategy must follow the study design and missingness mechanism.

### Duplicates

```python
df.duplicated().sum()
df.drop_duplicates()
```

### Impossible Values

Examples:

```python
df[df["Age"] < 0]
df[df["Fare"] < 0]
```

### Outliers

The notebook uses the IQR rule:

```text
IQR = Q3 − Q1
Lower fence = Q1 − 1.5 × IQR
Upper fence = Q3 + 1.5 × IQR
```

Outliers are **flagged first**, not automatically deleted.

---

# 8. `map()` and `apply()`

### `map()`

Useful for simple value-to-value mappings.

```python
df["survival_label"] = df["survived"].map({
    0: "Did not survive",
    1: "Survived"
})
```

### `apply()`

Useful when a custom function is needed.

```python
def age_category(age):
    if age < 18:
        return "Child"
    elif age < 60:
        return "Adult"
    return "Older adult"

df["age_group"] = df["age"].apply(age_category)
```

---

# 9. Grouping & Aggregation

```python
df.groupby("sex")["survived"].mean()
```

Multiple summaries:

```python
df.groupby("pclass").agg(
    passengers=("passengerid", "count"),
    survival_rate=("survived", "mean"),
    mean_age=("age", "mean"),
    median_fare=("fare", "median")
)
```

---

# 10. Merge, Join & Concatenate

### Merge

```python
left.merge(right, on="passengerid", how="left")
```

### Join

```python
left.join(right)
```

### Concatenate

```python
pd.concat([df1, df2], axis=0)
```

The notebook creates small teaching tables from Titanic data so every example can run without an extra dataset.

---

# 11. Reshaping & Pivot Tables

### Pivot table

```python
pd.pivot_table(
    df,
    values="survived",
    index="sex",
    columns="pclass",
    aggfunc="mean"
)
```

### Wide → Long

```python
pd.melt(...)
```

---

# 12. Descriptive Statistics with Python

Core statistics covered:

- count;
- frequency;
- percentage;
- mean;
- median;
- mode;
- variance;
- standard deviation;
- minimum / maximum;
- range;
- quartiles;
- IQR;
- skewness.

### Numerical Summary

```python
df["age"].describe()
```

### Categorical Summary

```python
df["sex"].value_counts()
df["sex"].value_counts(normalize=True) * 100
```

---

# 13. Data Visualization

The notebook uses all three:

### Matplotlib
Best for learning the foundations and detailed figure control.

### Seaborn
Convenient for statistical graphics built on Matplotlib.

### Plotly
Useful for interactive figures.

Plots covered:

- histogram;
- bar plot;
- count plot;
- box plot;
- scatter plot;
- line plot;
- heatmap;
- pair plot;
- multiple panels / subplots;
- interactive Plotly charts.

---

# 14. Publication-Quality Figure Workflow

A simple publication workflow:

```text
Choose the research question
        ↓
Choose the correct plot
        ↓
Use readable labels
        ↓
Avoid unnecessary visual clutter
        ↓
Use appropriate figure dimensions
        ↓
Check axis scales
        ↓
Add informative caption/title
        ↓
Save at high resolution
```

Example:

```python
plt.tight_layout()
plt.savefig("figure_survival_by_class.png",
            dpi=300,
            bbox_inches="tight")
```

---

# 15. Exploratory Data Analysis (EDA)

EDA is the systematic process of understanding data before formal modeling.

### Univariate Analysis

One variable at a time:

- age distribution;
- fare distribution;
- passenger class frequencies;
- survival frequencies.

### Bivariate Analysis

Two variables:

- sex vs survival;
- class vs survival;
- age vs survival;
- fare vs class.

### Multivariate Analysis

Multiple variables simultaneously:

- survival by sex and class;
- age + fare + class + survival;
- grouped summaries;
- pair plots.

---

# 16. Correlation

```python
df[numeric_columns].corr()
```

> Correlation describes association, not causation.

---

# 17. Cross-Tabulation

```python
pd.crosstab(df["sex"], df["survived"])
```

Row percentages:

```python
pd.crosstab(
    df["sex"],
    df["survived"],
    normalize="index"
) * 100
```

---

# 18. Automated EDA

The notebook includes a simple reusable function:

```python
quick_eda(df)
```

It produces:

- shape;
- data types;
- unique counts;
- missing counts and percentages;
- duplicates;
- numeric descriptive statistics.

This is intentionally transparent so beginners can understand what the automation is doing.

---

# 19. Research-Oriented EDA Checklist

Before reporting or modeling, ask:

- What is the unit of observation?
- Is the outcome coded correctly?
- Are categorical labels consistent?
- Are variable types appropriate?
- How much data are missing?
- Are duplicates present?
- Are numeric values plausible?
- Are there influential outliers?
- What do distributions look like?
- How do key variables relate to the outcome?
- Are important subgroups very small?
- Could transformations or recoding be needed?
- Is there evidence of data leakage for future ML work?

---

# 20. Basic Inferential Statistics Preview

The final section introduces three common ideas.

### 95% Confidence Interval

A CI provides a range of plausible values for an estimated population quantity under the assumptions of the method.

### Welch Independent-Samples t-test

Example question:

> Is mean age different between passengers who survived and those who did not?

### Chi-Square Test of Independence

Example question:

> Is survival associated with sex?

These are included as a **preview**. A later statistics class should cover assumptions, effect sizes and interpretation more deeply.

---

# 21. Recommended Learning Resources

Use these as companion references.

### Python
- Python Tutorial: https://docs.python.org/3/tutorial/
- Kaggle Learn Python: https://www.kaggle.com/learn/python

### NumPy
- NumPy Learn: https://numpy.org/learn/
- NumPy Absolute Basics: https://numpy.org/doc/stable/user/absolute_beginners.html

### pandas
- pandas Getting Started: https://pandas.pydata.org/getting_started.html
- pandas User Guide: https://pandas.pydata.org/docs/user_guide/

### Matplotlib
- Matplotlib Tutorials: https://matplotlib.org/stable/tutorials/

### Seaborn
- Seaborn Tutorial: https://seaborn.pydata.org/tutorial.html

### Plotly
- Plotly Python: https://plotly.com/python/

### Google Colab
- Colab Welcome Notebook: https://colab.research.google.com/notebooks/welcome.ipynb

---

# 22. Suggested Homework

Using the same Titanic dataset:

1. Find the number and percentage of passengers in each class.
2. Calculate mean and median age by sex.
3. Calculate survival percentage by passenger class.
4. Create a new `family_size` variable.
5. Create an `is_alone` variable.
6. Identify missing values.
7. Identify duplicated records.
8. Flag fare outliers using the IQR method.
9. Create one histogram.
10. Create one box plot.
11. Create one count plot.
12. Create a survival-rate table by sex and class.
13. Create a correlation heatmap.
14. Save at least one figure at 300 dpi.
15. Export your cleaned dataset as CSV.

---

# ✅ Class 02 Checklist

```text
✓ NumPy arrays
✓ Dimensions
✓ Indexing & slicing
✓ Array operations
✓ Broadcasting
✓ Vectorization
✓ Statistics
✓ Linear algebra
✓ Random numbers
✓ pandas Series & DataFrame
✓ CSV / Excel / JSON / Stata
✓ Dataset audit
✓ Selecting / filtering
✓ Sorting / ranking
✓ Missing data
✓ Duplicates
✓ Impossible values
✓ Outlier flags
✓ map / apply
✓ GroupBy / aggregation
✓ Merge / join / concatenate
✓ Reshaping
✓ Pivot tables
✓ Matplotlib
✓ Seaborn
✓ Plotly
✓ Publication figures
✓ Descriptive statistics
✓ Uni / Bi / Multivariate EDA
✓ Correlation
✓ Cross-tabs
✓ Automated EDA
✓ Confidence interval
✓ t-test
✓ Chi-square
✓ Export clean data
```

---

# 🎓 Salek Data Lab

**Python for Data Science, Machine Learning & AI**

> Learn → Code → Analyze → Explain → Build → Deploy

