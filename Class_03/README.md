# Class 03 — Titanic Data Cleaning, Visualization & Statistics

**Course:** Python for Data Science, Machine Learning & AI  
**Class:** 03  
**Duration:** 3 hours  
**Dataset:** Titanic (891 rows × 12 original columns, in the commonly used training file)  
**Tools:** Google Colab, Python, Pandas, NumPy, Matplotlib, Plotly, SciPy, Statsmodels

---

## 1. Class overview

Class 03 takes students through a complete beginner-friendly data analysis workflow:

```text
RAW TITANIC DATA
      |
      v
LOAD & INSPECT
      |
      v
PANDAS DATA CLEANING
      |
      v
WRANGLING & FEATURE CREATION
      |
      v
MATPLOTLIB VISUALIZATION
      |
      v
INTERACTIVE PLOTLY VISUALIZATION
      |
      v
DESCRIPTIVE STATISTICS
      |
      v
CONFIDENCE INTERVAL & HYPOTHESIS TESTING
      |
      v
INTERPRET RESULTS & EXPORT CLEAN DATA
```

### Learning outcomes

After completing this class, students should be able to:

1. Inspect a dataset before making changes.
2. identify and handle missing values.
3. check duplicate records and invalid values.
4. transform variables and create new features.
5. summarize information with `groupby()`, `agg()`, `crosstab()` and `pivot_table()`.
6. create charts using Matplotlib.
7. create interactive charts using Plotly.
8. calculate descriptive statistics.
9. calculate a confidence interval for a proportion.
10. conduct and interpret a Welch independent-samples t-test.
11. conduct and interpret a chi-square test of independence.
12. explain why association does not establish causation.

---

## 2. Three-hour class plan

| Time | Section | Main contents |
|---|---|---|
| 00–10 min | Setup and inspection | Upload CSV, imports, shape, columns, types |
| 10–60 min | Pandas | Missing values, duplicates, invalid values, transformation, grouping, reshaping |
| 60–120 min | Visualization | Matplotlib and Plotly |
| 120–170 min | Statistics | Descriptive statistics, confidence interval, t-test, chi-square, Cramer's V |
| 170–180 min | Wrap-up | Interpretation, export and homework |

The notebook is deliberately practical. Students should run each code cell, inspect the output and explain what it means.

---

## 3. Files in this class

Recommended repository structure:

```text
python-data-science-course/
│
├── Class_01/
│
├── Class_02/
│
├── Class_03/
│   ├── Class_03_Titanic_Data_Cleaning_Visualization_Statistics.ipynb
│   ├── Class_03_Complete_Code.py
│   └── README.md
│
└── README.md
```

- **`.ipynb`** — the main Google Colab notebook. It contains explanations and executable code cells.
- **`.py`** — the complete Python code in script form. It is useful for reading or running in a Python environment, but the notebook is the recommended teaching format.
- **`README.md`** — this document. It explains the lesson, workflow, code and interpretation.

---

## 4. Dataset

The notebook expects a Titanic CSV file containing these 12 original columns:

| Column | Meaning |
|---|---|
| `PassengerId` | Unique passenger identifier |
| `Survived` | Survival outcome: 0 = did not survive, 1 = survived |
| `Pclass` | Ticket class: 1, 2 or 3 |
| `Name` | Passenger name |
| `Sex` | Passenger sex |
| `Age` | Passenger age |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin identifier |
| `Embarked` | Port of embarkation |

Dataset source: [Kaggle — Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

Download the CSV file and upload it when the first notebook cell asks for a file. The code accepts a CSV filename other than `Titanic-Dataset.csv`, provided that it has the required columns.

**Data note:** The familiar 891-row Titanic training dataset has 177 missing Age values, 687 missing Cabin values and 2 missing Embarked values. The notebook calculates the actual missing counts in the uploaded file instead of assuming those counts.

---

# 5. Code explained — section by section

The explanations below follow the notebook's execution order. Each code statement is explained in plain language, including important arguments and method calls.

## Section 0 — Upload the CSV in Google Colab

```python
from google.colab import files
uploaded = files.upload()
```

- `from google.colab import files`: imports Colab's file utility.
- `files.upload()`: opens a file picker in the notebook.
- `uploaded`: stores information about the uploaded files.

```python
csv_files = [name for name in uploaded.keys() if name.lower().endswith(".csv")]
```

- `uploaded.keys()`: gets the uploaded filenames.
- `name`: represents one filename at a time.
- `name.lower()`: converts the filename to lowercase, so `.CSV` and `.csv` are both accepted.
- `.endswith(".csv")`: checks whether the filename ends in `.csv`.
- The list comprehension creates a list containing only CSV filenames.

```python
if not csv_files:
    raise FileNotFoundError("Please upload a Titanic CSV file.")
```

- `if not csv_files`: checks whether the list is empty.
- `raise FileNotFoundError(...)`: stops execution and displays a useful error if no CSV was uploaded.

```python
file_path = csv_files[0]
print("Reading:", file_path)
```

- `[0]` selects the first item in the list. Python indexing starts at zero.
- `file_path` stores the selected filename.
- `print()` displays the filename.

**Why this matters:** Students do not need to rename the downloaded file to one exact name.

---

## Section 1 — Import libraries

```python
import pandas as pd
```

Imports Pandas and gives it the short name `pd`. Pandas is used to work with rows and columns.

```python
import numpy as np
```

Imports NumPy as `np`. It provides numerical operations and tools such as `np.where()` and `np.inf`.

```python
import matplotlib.pyplot as plt
```

Imports Matplotlib's plotting interface as `plt`. It is used to create static charts.

```python
import plotly.express as px
import plotly.graph_objects as go
```

- `plotly.express` (`px`) provides a concise way to create interactive charts.
- `plotly.graph_objects` (`go`) provides lower-level chart components. It is imported as a useful companion library, although the main lesson uses Plotly Express.

```python
from scipy import stats
```

Imports SciPy's statistical functions. The notebook uses `stats.ttest_ind()` and `stats.chi2_contingency()`.

```python
from statsmodels.stats.proportion import proportion_confint
```

Imports a function for calculating confidence intervals for proportions.

```python
from IPython.display import display
```

Imports `display()` to show tables and DataFrames clearly inside Jupyter or Colab.

```python
pd.set_option("display.max_columns", None)
```

Tells Pandas to display all columns instead of hiding some columns.

```python
pd.set_option("display.float_format", "{:,.3f}".format)
```

Displays floating-point values with three digits after the decimal point. This changes display formatting, not the underlying stored values.

```python
RANDOM_STATE = 42
```

Stores a fixed number for reproducible sampling. The number 42 has no special statistical meaning.

---

## Section 2 — Read and inspect the dataset

```python
df_raw = pd.read_csv(file_path)
```

- `pd.read_csv()` reads a CSV file.
- `file_path` tells Pandas which file to read.
- `df_raw` stores the resulting DataFrame.
- The suffix `_raw` reminds us that this is the original, uncleaned copy.

```python
print("Dataset shape:", df_raw.shape)
```

- `.shape` returns a pair: `(number_of_rows, number_of_columns)`.
- For the commonly used Titanic training dataset, the expected result is `(891, 12)`.

```python
display(df_raw.head())
```

- `.head()` displays the first five rows by default.
- It helps students understand the structure of the data.

```python
display(df_raw.sample(min(5, len(df_raw)), random_state=RANDOM_STATE))
```

- `.sample()` selects random rows.
- `len(df_raw)` returns the number of rows.
- `min(5, len(df_raw))` ensures that the requested sample size does not exceed the dataset size.
- `random_state=42` makes the selected sample reproducible.

```python
print(df_raw.columns.tolist())
```

- `.columns` returns the column labels.
- `.tolist()` converts them into a normal Python list.

```python
df_raw.info()
```

Displays the number of rows, column names, non-null counts, data types and approximate memory use.

```python
display(df_raw.describe(include="all").T)
```

- `.describe()` calculates summary statistics.
- `include="all"` includes both numeric and non-numeric columns.
- `.T` transposes the output so each variable appears on a separate row.

**Teaching point:** Inspection comes before cleaning. Do not change data before understanding its structure.

---

## Section 3 — Preserve the original and standardize column names

```python
df = df_raw.copy()
```

Creates a separate working copy. Cleaning `df` will not modify `df_raw`.

```python
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(r"[^a-z0-9]+", "_", regex=True)
      .str.strip("_")
)
```

Line by line:

- `df.columns`: accesses the column names.
- `.str`: applies string operations to all column names.
- `.strip()`: removes spaces from the beginning and end.
- `.lower()`: converts names to lowercase.
- `.str.replace(...)`: replaces characters that are not lowercase letters or digits with underscores.
- `r"[^a-z0-9]+"`: a regular expression matching one or more characters outside the allowed set.
- `regex=True`: tells Pandas to interpret the pattern as a regular expression.
- The final `.str.strip("_")` removes unnecessary underscores from the beginning and end.

For example, `PassengerId` becomes `passengerid`, and `Passenger Name` would become `passenger_name`.

---

## Section 4 — Validate required columns

```python
required_columns = {
    "passengerid", "survived", "pclass", "name", "sex", "age",
    "sibsp", "parch", "ticket", "fare", "cabin", "embarked"
}
```

Creates a set of column names that the notebook needs.

```python
missing_columns = required_columns.difference(df.columns)
```

- `.difference()` returns the required column names that are absent from the uploaded dataset.

```python
if missing_columns:
    raise ValueError(...)
```

Stops execution if one or more required columns are missing. This avoids confusing errors later in the notebook.

---

## Section 5 — Audit missing values and duplicates

```python
missing_audit = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2)
}).sort_values("missing_percent", ascending=False)
```

Breakdown:

- `df.isna()`: creates a True/False table; True means the cell is missing.
- `.sum()`: counts True values in each column.
- `.mean()`: calculates the proportion of missing values in each column because True is treated as 1 and False as 0.
- `* 100`: converts proportions into percentages.
- `.round(2)`: rounds percentages to two decimal places.
- `pd.DataFrame({...})`: combines the two results into one table.
- `.sort_values(...)`: sorts the table by missing percentage.
- `ascending=False`: places the highest missing percentage first.

```python
df.duplicated().sum()
```

- `.duplicated()` identifies rows that duplicate an earlier row.
- `.sum()` counts those rows.

```python
df["passengerid"].duplicated().sum()
```

Checks whether the passenger identifier appears more than once. This is a different check from exact duplicate rows.

---

## Section 6 — Treat missing values

```python
df["age_was_missing"] = df["age"].isna()
df["cabin_was_missing"] = df["cabin"].isna()
```

Creates two Boolean indicator columns before filling missing values.

- `True`: the original value was missing.
- `False`: the original value was present.

This preserves information about missingness.

```python
age_group_medians = df.groupby(["pclass", "sex"])["age"].transform("median")
```

- `.groupby(["pclass", "sex"])`: divides passengers into groups based on passenger class and sex.
- `["age"]`: selects the Age variable.
- `.transform("median")`: calculates each group's median age and returns a result aligned with the original rows.
- `age_group_medians`: stores the median appropriate to each passenger's group.

```python
df["age"] = df["age"].fillna(age_group_medians)
```

Fills missing Age values using the median for that passenger's class and sex group.

```python
df["age"] = df["age"].fillna(df["age"].median())
```

Uses the overall median as a fallback if a group median was unavailable.

```python
embarked_mode = df["embarked"].mode(dropna=True)
```

- `.mode()` finds the most frequent category.
- `dropna=True` excludes missing values when finding the mode.

```python
if not embarked_mode.empty:
    df["embarked"] = df["embarked"].fillna(embarked_mode.iloc[0])
```

- `.empty` checks whether the mode result contains any values.
- `.iloc[0]` selects the first mode.
- `.fillna()` replaces missing Embarked values with that mode.

```python
df["cabin"] = df["cabin"].fillna("Unknown")
```

Replaces missing cabin text with the label `Unknown`. This does not claim to know the passenger's cabin. The earlier indicator column preserves whether the original cabin was missing.

**Why the strategies differ:** Age is numeric, Embarked is categorical with very few missing values, and Cabin has a large amount of missing information. One missing-value strategy is not suitable for every variable.

---

## Section 7 — Check types and impossible values

```python
df.dtypes
```

Displays the data type of each column.

```python
df[df["age"] < 0]
df[df["fare"] < 0]
```

Selects rows with negative age or fare. These are examples of values that need investigation.

```python
df[~df["pclass"].isin([1, 2, 3])]
```

- `.isin([1, 2, 3])` checks whether each passenger class is one of the permitted values.
- `~` reverses the Boolean result.
- The expression selects unexpected passenger classes.

```python
df[~df["survived"].isin([0, 1])]
```

Checks for survival codes other than 0 and 1.

**Important:** The code reports suspicious records. It does not automatically delete them.

---

## Section 8 — Remove exact duplicates

```python
df = df.drop_duplicates().copy()
```

- `.drop_duplicates()` removes exact duplicate rows.
- `.copy()` creates a clean working copy of the result.

Do not remove passengers merely because they share a name or ticket number. Those are not necessarily duplicate records.

---

## Section 9 — Create new variables

### Family size

```python
df["family_size"] = 1 + df["sibsp"] + df["parch"]
```

Counts the passenger plus the listed siblings/spouses and parents/children aboard.

Example: if `SibSp = 1` and `Parch = 2`, family size is `1 + 1 + 2 = 4`.

### Travelling alone

```python
df["is_alone"] = np.where(df["family_size"] == 1, "Alone", "With family")
```

- `np.where(condition, value_if_true, value_if_false)` selects a label based on a condition.
- If family size equals 1, the passenger is labelled Alone.
- Otherwise, the passenger is labelled With family.

### Extract title from name

```python
df["title"] = df["name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
```

- `.str.extract()` extracts text matching a regular expression.
- `r",\s*([^.]*)\."` looks for text after a comma and before the next period.
- `\s*` allows optional spaces.
- `([^.]*)` captures characters up to a period.
- `expand=False` returns a Series rather than a DataFrame.
- `.str.strip()` removes extra spaces.

Example: `Braund, Mr. Owen Harris` produces `Mr`.

### Extract deck

```python
df["deck"] = df["cabin"].where(df["cabin"] != "Unknown").str[0]
```

- `.where(condition)` keeps values where the condition is True.
- Unknown cabin entries become missing in the deck variable.
- `.str[0]` extracts the first character of the cabin code.

### Survival label

```python
df["survival_label"] = df["survived"].map({
    0: "Did not survive",
    1: "Survived"
})
```

Maps the numeric codes into readable categories. The original `survived` variable remains unchanged.

### Age group

```python
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, np.inf],
    labels=["Child", "Teenager", "Young adult", "Adult", "Older adult"],
    include_lowest=True
)
```

- `pd.cut()` divides a continuous numeric variable into intervals.
- `bins` defines the interval boundaries.
- `np.inf` represents an unlimited upper boundary.
- `labels` names the intervals.
- `include_lowest=True` includes the lowest boundary in the first interval.

These are teaching categories, not clinical or official age classifications.

---

## Section 10 — Map and standardize values

```python
df["pclass_label"] = df["pclass"].map({
    1: "First class",
    2: "Second class",
    3: "Third class"
})
```

Uses a dictionary to convert class codes into readable labels.

```python
df["sex"] = df["sex"].astype("string").str.strip().str.lower()
```

- `.astype("string")` converts the column to Pandas' string data type.
- `.str.strip()` removes surrounding spaces.
- `.str.lower()` converts text to lowercase.

```python
df["embarked"] = df["embarked"].astype("string").str.strip().str.upper()
```

Converts Embarked values to uppercase and removes extra spaces.

---

## Section 11 — Select, filter and sort

```python
df[["passengerid", "sex", "age", "pclass", "fare", "survived"]]
```

Selects several columns. A list of column names inside double brackets returns a DataFrame.

```python
df.loc[df["age"] >= 18]
```

Selects passengers aged 18 or older. `.loc` is label-based selection.

```python
df.loc[
    (df["sex"] == "female") & (df["pclass"] == 1)
]
```

Selects passengers who satisfy both conditions.

- `&` means AND for Pandas Boolean conditions.
- Each condition is placed in parentheses.

```python
df.sort_values("fare", ascending=False)
```

Sorts passengers by fare, from highest to lowest.

```python
df.iloc[:5, :4]
```

Selects the first five rows and first four columns by integer position. The ending position is excluded.

---

## Section 12 — Grouping and aggregation

```python
df.groupby("sex", observed=False)["survived"].agg(
    passengers="count",
    survival_rate="mean"
)
```

- `.groupby("sex")`: splits passengers into sex groups.
- `["survived"]`: selects the survival variable.
- `.agg()`: calculates multiple summaries.
- `passengers="count"`: counts non-missing survival observations.
- `survival_rate="mean"`: calculates the mean of the 0/1 survival variable.

Because survival is coded 0 and 1, its mean is the proportion who survived.

```python
df.groupby("pclass", observed=False).agg(
    passengers=("passengerid", "count"),
    survival_rate=("survived", "mean"),
    mean_age=("age", "mean"),
    median_fare=("fare", "median")
)
```

This is named aggregation:

- `passengers`: count PassengerId values.
- `survival_rate`: calculate mean survival.
- `mean_age`: calculate average age.
- `median_fare`: calculate median fare.

`observed=False` controls whether unobserved categorical groups are included.

---

## Section 13 — Cross-tabulation and pivot table

```python
pd.crosstab(df["sex"], df["survived"])
```

Counts the combinations of sex and survival outcome.

```python
pd.crosstab(
    df["sex"], df["survived"], normalize="index"
) * 100
```

- `normalize="index"` converts each row into proportions.
- `* 100` converts proportions into percentages.

```python
pd.pivot_table(
    df,
    values="survived",
    index="sex",
    columns="pclass",
    aggfunc="mean",
    observed=False
) * 100
```

- `values`: variable to summarize.
- `index`: variables shown in rows.
- `columns`: variables shown in columns.
- `aggfunc="mean"`: calculates mean survival.
- Multiplication by 100 expresses survival as a percentage.

---

## Section 14 — Reshape wide data to long data

```python
class_summary = (
    df.groupby("pclass", observed=False)
      .agg(mean_age=("age", "mean"), median_fare=("fare", "median"))
      .reset_index()
)
```

Creates a summary table by passenger class.

- `.reset_index()` converts the grouped index back into a regular column.

```python
class_summary_long = class_summary.melt(
    id_vars="pclass",
    var_name="measure",
    value_name="value"
)
```

- `.melt()` converts wide-format columns into rows.
- `id_vars="pclass"` keeps passenger class as an identifier.
- `var_name="measure"` names the column that stores former column names.
- `value_name="value"` names the column that stores the corresponding values.

---

## Section 15 — Outlier flagging with IQR

```python
q1 = df["fare"].quantile(0.25)
q3 = df["fare"].quantile(0.75)
```

Calculates the first quartile (25th percentile) and third quartile (75th percentile).

```python
iqr = q3 - q1
```

Calculates the interquartile range.

```python
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
```

Calculates the conventional IQR outlier fences.

```python
df["fare_outlier_flag"] = (
    (df["fare"] < lower_fence) | (df["fare"] > upper_fence)
)
```

- `< lower_fence`: checks unusually low fares.
- `> upper_fence`: checks unusually high fares.
- `|` means OR.
- The result is stored as a Boolean flag.

**Interpretation:** A flagged fare is a possible outlier, not automatically an error. Do not delete it without a defensible reason.

---

# 6. Visualization explained

## Section 16 — Matplotlib histogram

```python
plt.figure(figsize=(9, 5))
```

Creates a figure 9 inches wide and 5 inches high.

```python
plt.hist(df["age"].dropna(), bins=20, edgecolor="black")
```

- `plt.hist()` creates a histogram.
- `dropna()` excludes missing values.
- `bins=20` divides the age range into 20 intervals.
- `edgecolor="black"` outlines the bars.

```python
plt.title(...)
plt.xlabel(...)
plt.ylabel(...)
```

Adds a chart title and axis labels.

```python
plt.grid(axis="y", alpha=0.25)
```

Adds horizontal grid lines with low opacity.

```python
plt.tight_layout()
plt.show()
```

Adjusts spacing and displays the chart.

## Section 17 — Matplotlib bar chart

```python
survival_counts_overall = df["survived"].value_counts().sort_index()
```

- `.value_counts()` counts each outcome.
- `.sort_index()` orders the outcome codes as 0 and 1.

```python
plt.bar(["Did not survive", "Survived"], survival_counts_overall.values, edgecolor="black")
```

Creates a bar chart. The first argument provides category labels; the second provides bar heights.

## Section 18 — Survival rate by class

```python
class_survival = df.groupby("pclass", observed=False)["survived"].mean() * 100
```

Calculates the mean survival outcome in each class and converts it to a percentage.

```python
plt.ylim(0, 100)
```

Fixes the vertical axis from 0% to 100%, making the scale easy to interpret.

## Section 19 — Box plot

```python
fare_groups = [
    df.loc[df["pclass"] == class_no, "fare"].dropna()
    for class_no in [1, 2, 3]
]
```

Creates three fare groups, one for each passenger class.

```python
plt.boxplot(fare_groups, labels=["First", "Second", "Third"])
```

Draws a box plot for each group.

A box plot displays the median, quartiles, spread and potential outliers.

## Section 20 — Scatter plot

```python
plt.scatter(df["age"], df["fare"], alpha=0.55)
```

Plots one point per passenger:

- x-axis: age.
- y-axis: fare.
- `alpha=0.55`: makes points partly transparent so overlapping observations are easier to see.

## Section 21 — Correlation heatmap

```python
numeric_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare", "family_size"]
correlation_matrix = df[numeric_columns].corr()
```

Selects numeric variables and calculates pairwise Pearson correlations.

```python
plt.imshow(correlation_matrix, aspect="auto", vmin=-1, vmax=1)
```

Displays the correlation matrix as a color-coded image.

- `vmin=-1` and `vmax=1` set the correlation scale.
- Correlations range from -1 to +1.

The nested `for` loops place each correlation coefficient inside its heatmap cell.

**Interpretation:**
- Positive value: variables tend to increase together.
- Negative value: one tends to decrease as the other increases.
- Near zero: weak linear association.

Correlation does not establish causation.

## Section 22 — Plotly grouped bar chart

```python
plot_df = (
    df.groupby(["sex", "pclass"], observed=False)["survived"]
      .mean()
      .reset_index()
)
```

Calculates survival proportions for every sex and class combination.

```python
plot_df["survival_percent"] = plot_df["survived"] * 100
plot_df["pclass"] = plot_df["pclass"].astype(str)
```

Converts the survival proportion to a percentage and class codes to strings for categorical plotting.

```python
fig = px.bar(
    plot_df,
    x="sex",
    y="survival_percent",
    color="pclass",
    barmode="group",
    text="survival_percent",
    title="Survival Rate by Sex and Passenger Class"
)
```

- `px.bar()`: creates an interactive bar chart.
- `x`: category on the horizontal axis.
- `y`: numeric measure on the vertical axis.
- `color`: creates separate color groups.
- `barmode="group"`: places group bars side by side.
- `text`: adds values to bars.
- `title`: sets the chart title.

```python
fig.update_traces(...)
fig.update_layout(...)
fig.show()
```

Updates the bar labels, fixes the y-axis range and displays the interactive chart.

## Section 23 — Plotly histogram

```python
fig = px.histogram(
    df,
    x="age",
    color="survival_label",
    nbins=25,
    barmode="overlay",
    opacity=0.65
)
```

- `x="age"`: displays age distribution.
- `color="survival_label"`: separates survivors and non-survivors.
- `nbins=25`: requests 25 histogram bins.
- `barmode="overlay"`: overlays the groups.
- `opacity=0.65`: makes overlapping bars easier to inspect.

## Section 24 — Family size and survival

```python
family_survival = (
    df.groupby("family_size", observed=False)["survived"]
      .agg(passengers="count", survival_rate="mean")
      .reset_index()
)
```

Calculates passenger counts and survival proportions for each family size.

The passenger count is included as hover information because groups with very few passengers can produce unstable percentages.

---

# 7. Statistics explained

## Section 25 — Numerical descriptive statistics

```python
df[["age", "fare", "family_size"]].describe().T
```

Calculates:

| Statistic | Meaning |
|---|---|
| `count` | Number of non-missing observations |
| `mean` | Arithmetic average |
| `std` | Sample standard deviation |
| `min` | Smallest value |
| `25%` | First quartile |
| `50%` | Median |
| `75%` | Third quartile |
| `max` | Largest value |

```python
df["age"].mean()
df["age"].median()
df["age"].std()
```

Calculates the mean, median and sample standard deviation of age.

## Section 26 — Categorical descriptive statistics

```python
df[column].value_counts(dropna=False)
```

Counts the frequency of each category, including missing values.

```python
df[column].value_counts(normalize=True, dropna=False) * 100
```

Calculates percentages rather than raw counts.

- `normalize=True`: returns proportions.
- `dropna=False`: includes missing values as a category.
- `* 100`: converts proportions to percentages.

## Section 27 — 95% confidence interval for survival proportion

```python
survived_count = int(df["survived"].sum())
total_count = int(df["survived"].count())
survival_proportion = survived_count / total_count
```

- `sum()` counts survivors because the outcome is coded 1 for survived and 0 otherwise.
- `count()` counts non-missing survival observations.
- Dividing gives the observed survival proportion.

```python
ci_low, ci_high = proportion_confint(
    count=survived_count,
    nobs=total_count,
    alpha=0.05,
    method="wilson"
)
```

- `count`: number of observed successes.
- `nobs`: total number of observations.
- `alpha=0.05`: requests a 95% confidence interval.
- `method="wilson"`: uses the Wilson score interval.

The output gives lower and upper confidence limits.

**Interpretation:** Under the assumptions of the sampling model, the interval is a plausible range for the population survival proportion. A confidence interval does not remove sampling bias or guarantee that the dataset represents every passenger population.

## Section 28 — Welch independent-samples t-test

**Research question:** Is mean age different between survivors and non-survivors?

```python
age_survived = df.loc[df["survived"] == 1, "age"].dropna()
age_not_survived = df.loc[df["survived"] == 0, "age"].dropna()
```

Creates two independent age samples.

```python
t_stat, p_value = stats.ttest_ind(
    age_survived,
    age_not_survived,
    equal_var=False,
    alternative="two-sided"
)
```

- `ttest_ind()`: performs an independent-samples t-test.
- `equal_var=False`: uses Welch's version, which does not require equal population variances.
- `alternative="two-sided"`: tests for a difference in either direction.
- `t_stat`: calculated t statistic.
- `p_value`: probability, under the null model, of observing a result at least as extreme as the one obtained.

Hypotheses:

- **H0:** Mean age is equal between survivors and non-survivors.
- **H1:** Mean age differs between the groups.

Decision rule at a 5% significance level:

| Result | Decision |
|---|---|
| p-value < 0.05 | Reject H0 |
| p-value ≥ 0.05 | Fail to reject H0 |

Do not write "accept H0." A non-significant result means there is insufficient evidence to reject it; it does not prove equality.

## Section 29 — Chi-square test of independence

**Research question:** Is sex associated with survival status?

```python
contingency_table = pd.crosstab(df["sex"], df["survived"])
```

Creates a frequency table showing the number of passengers in each sex-survival combination.

```python
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency_table)
```

Returns:

- `chi2`: chi-square test statistic.
- `chi_p`: p-value.
- `dof`: degrees of freedom.
- `expected`: expected cell counts if the variables were independent.

Hypotheses:

- **H0:** Sex and survival status are independent.
- **H1:** Sex and survival status are associated.

```python
expected < 5
```

Checks whether any expected cell count is below 5. Very small expected counts can make the usual chi-square approximation unreliable.

## Section 30 — Cramer's V

```python
n = contingency_table.to_numpy().sum()
r, k = contingency_table.shape
cramers_v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))
```

- `.to_numpy()` converts the table into a NumPy array.
- `.sum()` calculates the total sample size.
- `.shape` returns the number of rows and columns.
- `min(r - 1, k - 1)` adjusts the denominator for table dimensions.
- `np.sqrt()` calculates the square root.

Cramer's V ranges from 0 to 1. It describes association strength; it is not a p-value. Interpret its magnitude in context rather than relying on universal cutoffs.

## Section 31 — Pearson correlation

```python
df[numeric_columns].corr(method="pearson")
```

Calculates pairwise Pearson correlation coefficients for the selected numeric variables.

Pearson correlation measures linear association. It can be affected by outliers and does not establish causation.

---

## Section 32 — Final data audit

```python
df.shape
df.isna().sum()
df.duplicated().sum()
df.dtypes
```

These checks confirm the final dataset's dimensions, remaining missing values, exact duplicate rows and data types.

The audit is important because cleaning should be verified, not merely performed.

## Section 33 — Export cleaned data and a figure

```python
df.to_csv("titanic_cleaned_class03.csv", index=False)
```

Exports the cleaned DataFrame as a CSV file.

- `index=False`: prevents Pandas from adding its row index as an extra column.

```python
plt.savefig("survival_by_class_class03.png", dpi=300, bbox_inches="tight")
```

Saves the figure.

- `dpi=300`: exports at high resolution.
- `bbox_inches="tight"`: trims unnecessary surrounding whitespace.

```python
files.download(cleaned_file)
files.download(figure_file)
```

Downloads the exported files from Google Colab to the user's computer.

---

# 8. Important interpretation rules

Students should remember these distinctions:

| Concept | Correct interpretation | Incorrect interpretation |
|---|---|---|
| Mean of a 0/1 variable | Proportion of 1s | Average category label |
| p-value | Evidence against the null model, measured by the test | Probability that H0 is true |
| 95% confidence interval | Interval procedure with 95% long-run coverage under assumptions | 95% probability that this fixed interval contains the parameter |
| Correlation | Strength and direction of linear association | Proof that one variable causes another |
| Outlier flag | Observation requiring investigation | Automatic data error |
| Missing value | Information not recorded or unavailable | Always safe to delete |
| Statistical significance | Evidence relative to a chosen threshold | Proof of practical importance |

---

# 9. Research questions students can answer

1. What percentage of passengers survived?
2. Which passenger class had the highest observed survival rate?
3. How did observed survival rates differ by sex?
4. Was mean age different between survivors and non-survivors?
5. Was sex associated with survival status?
6. How did fare distributions differ across passenger classes?
7. Did family size show a pattern with survival?
8. Which variables had the strongest positive or negative linear correlations with survival?

**Caution:** These are observational data. Results should be described as patterns, differences or associations—not causal effects.

---

# 10. Homework

Use the same Titanic CSV:

1. Report missing counts and percentages before cleaning.
2. Explain the different missing-value strategies for Age, Embarked and Cabin.
3. Create `family_size` and `is_alone`.
4. Calculate survival percentage by sex and class.
5. Create a Matplotlib histogram and box plot.
6. Create an interactive Plotly chart.
7. Calculate a 95% confidence interval for survival proportion.
8. Perform a Welch t-test comparing age by survival status.
9. Perform a chi-square test for sex and survival.
10. Report Cramer's V.
11. Write a short interpretation of the results.
12. Export the cleaned dataset as CSV.

---

# 11. How to run the notebook in Google Colab

### Option A — Upload the notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Upload notebook**.
3. Upload `Class_03_Titanic_Data_Cleaning_Visualization_Statistics.ipynb`.
4. Run the first code cell.
5. Upload the Titanic CSV when prompted.
6. Run the remaining cells in order.

### Option B — Open from GitHub

After pushing the notebook to the repository, open:

```text
https://colab.research.google.com/github/muhammadsalek/python-data-science-course/blob/main/Class_03/Class_03_Titanic_Data_Cleaning_Visualization_Statistics.ipynb
```

If your GitHub username or repository name changes, update the URL accordingly.

### Git commands

Run these commands from the root of the repository:

```bash
git add Class_03/
git commit -m "Add Class 03 Titanic data analysis"
git push origin main
```

---

## 12. Recommended learning resources

- [Pandas Getting Started](https://pandas.pydata.org/getting_started.html)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [SciPy Statistics](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Statsmodels Proportion Tests and Confidence Intervals](https://www.statsmodels.org/stable/stats.html)
- [Google Colab](https://colab.research.google.com/)

---

**Salek Data Lab**  
*Python for Data Science, Machine Learning & AI*  
**Learn → Code → Analyze → Explain → Build → Deploy**

