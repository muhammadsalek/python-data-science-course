
# ==============================================================================
# Class 03 — Data Cleaning, Visualization & Statistics with Python
# ==============================================================================


# ==============================================================================
# 0. Load the Titanic CSV into Google Colab
# ==============================================================================

from google.colab import files
uploaded = files.upload()

# Find the first uploaded CSV file.
csv_files = [name for name in uploaded.keys() if name.lower().endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("Please upload a Titanic CSV file.")

file_path = csv_files[0]
print("Reading:", file_path)


# ==============================================================================
# 1. Import libraries
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from IPython.display import display

# Make Pandas output easier to inspect.
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:,.3f}".format)

# Reproducible random seed for any sampling used in this notebook.
RANDOM_STATE = 42


# ==============================================================================
# 2. Read and inspect the original data
# ==============================================================================

df_raw = pd.read_csv(file_path)

print("Dataset shape:", df_raw.shape)
display(df_raw.head())
display(df_raw.sample(min(5, len(df_raw)), random_state=RANDOM_STATE))

print("Column names:")
print(df_raw.columns.tolist())

print("\\nData types and non-null counts:")
df_raw.info()

print("\\nDescriptive summary:")
display(df_raw.describe(include="all").T)


# ==============================================================================
# PART 1 — PANDAS: DATA CLEANING, WRANGLING & TRANSFORMATION
# ==============================================================================

df = df_raw.copy()

# Remove accidental spaces, lowercase names, and replace spaces with underscores.
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(r"[^a-z0-9]+", "_", regex=True)
      .str.strip("_")
)

print(df.columns.tolist())


# ==============================================================================
# 4. Check whether the required Titanic columns exist
# ==============================================================================

required_columns = {
    "passengerid", "survived", "pclass", "name", "sex", "age",
    "sibsp", "parch", "ticket", "fare", "cabin", "embarked"
}

missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(f"The uploaded file is missing required columns: {sorted(missing_columns)}")

print("All required Titanic columns are present.")


# ==============================================================================
# 5. Audit missing values and duplicates
# ==============================================================================

missing_audit = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2)
}).sort_values("missing_percent", ascending=False)

display(missing_audit)
print("Exact duplicate rows:", df.duplicated().sum())
print("Duplicate PassengerId values:", df["passengerid"].duplicated().sum())


# ==============================================================================
# 6. Understand the missing-data pattern
# ==============================================================================

# Record missingness before filling values.
df["age_was_missing"] = df["age"].isna()
df["cabin_was_missing"] = df["cabin"].isna()

# Age: group median by passenger class and sex.
age_group_medians = df.groupby(["pclass", "sex"])["age"].transform("median")
df["age"] = df["age"].fillna(age_group_medians)

# If a group has no observed age, use the overall median.
df["age"] = df["age"].fillna(df["age"].median())

# Embarked: use the most frequent category.
embarked_mode = df["embarked"].mode(dropna=True)
if not embarked_mode.empty:
    df["embarked"] = df["embarked"].fillna(embarked_mode.iloc[0])

# For descriptive work, make missing cabin text explicit.
df["cabin"] = df["cabin"].fillna("Unknown")

print("Missing values after the selected treatment:")
display(df.isna().sum().to_frame("missing_count"))


# ==============================================================================
# 7. Check data types and impossible values
# ==============================================================================

print(df.dtypes)

invalid_age = df[df["age"] < 0]
invalid_fare = df[df["fare"] < 0]
invalid_pclass = df[~df["pclass"].isin([1, 2, 3])]
invalid_survived = df[~df["survived"].isin([0, 1])]

print("Negative ages:", len(invalid_age))
print("Negative fares:", len(invalid_fare))
print("Unexpected passenger classes:", len(invalid_pclass))
print("Unexpected survival codes:", len(invalid_survived))


# ==============================================================================
# 8. Remove exact duplicate rows only when justified
# ==============================================================================

df = df.drop_duplicates().copy()
print("Shape after removing exact duplicate rows:", df.shape)


# ==============================================================================
# 9. Create useful variables (feature engineering)
# ==============================================================================

df["family_size"] = 1 + df["sibsp"] + df["parch"]
df["is_alone"] = np.where(df["family_size"] == 1, "Alone", "With family")

# Extract the title between the comma and period in names such as "Braund, Mr. Owen Harris".
df["title"] = df["name"].str.extract(r",\\s*([^.]*)\\.", expand=False).str.strip()

# Cabin deck is the first character of the cabin field; Unknown remains missing as a deck.
df["deck"] = df["cabin"].where(df["cabin"] != "Unknown").str[0]

df["survival_label"] = df["survived"].map({
    0: "Did not survive",
    1: "Survived"
})

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, np.inf],
    labels=["Child", "Teenager", "Young adult", "Adult", "Older adult"],
    include_lowest=True
)

display(df[[
    "passengerid", "age", "age_group", "sex", "pclass",
    "family_size", "is_alone", "title", "deck", "survival_label"
]].head(10))


# ==============================================================================
# 10. Transform values with `map()` and `replace()`
# ==============================================================================

df["pclass_label"] = df["pclass"].map({
    1: "First class",
    2: "Second class",
    3: "Third class"
})

df["sex"] = df["sex"].astype("string").str.strip().str.lower()
df["embarked"] = df["embarked"].astype("string").str.strip().str.upper()

display(df[["pclass", "pclass_label", "sex", "embarked"]].head())


# ==============================================================================
# 11. Filter, select and sort records
# ==============================================================================

# Select a few columns.
display(df[["passengerid", "sex", "age", "pclass", "fare", "survived"]].head())

# Passengers aged 18 or older.
adults = df.loc[df["age"] >= 18]
print("Adult passengers:", len(adults))

# Female passengers in first class.
female_first_class = df.loc[
    (df["sex"] == "female") & (df["pclass"] == 1)
]
display(female_first_class.head())

# Sort by fare from highest to lowest.
display(df.sort_values("fare", ascending=False)[
    ["passengerid", "name", "pclass", "fare"]
].head(10))

# First five rows and first four columns by position.
display(df.iloc[:5, :4])


# ==============================================================================
# 12. Grouping and aggregation
# ==============================================================================

survival_by_sex = df.groupby("sex", observed=False)["survived"].agg(
    passengers="count",
    survival_rate="mean"
)
survival_by_sex["survival_percent"] = survival_by_sex["survival_rate"] * 100
display(survival_by_sex)

summary_by_class = df.groupby("pclass", observed=False).agg(
    passengers=("passengerid", "count"),
    survival_rate=("survived", "mean"),
    mean_age=("age", "mean"),
    median_fare=("fare", "median")
)
summary_by_class["survival_percent"] = summary_by_class["survival_rate"] * 100
display(summary_by_class.round(2))


# ==============================================================================
# 13. Cross-tabulation and pivot tables
# ==============================================================================

survival_counts = pd.crosstab(df["sex"], df["survived"])
display(survival_counts)

survival_row_percent = pd.crosstab(
    df["sex"], df["survived"], normalize="index"
) * 100
display(survival_row_percent.round(1))

survival_pivot = pd.pivot_table(
    df,
    values="survived",
    index="sex",
    columns="pclass",
    aggfunc="mean",
    observed=False
) * 100
display(survival_pivot.round(1))


# ==============================================================================
# 14. Reshape data from wide to long format
# ==============================================================================

class_summary = (
    df.groupby("pclass", observed=False)
      .agg(mean_age=("age", "mean"), median_fare=("fare", "median"))
      .reset_index()
)

class_summary_long = class_summary.melt(
    id_vars="pclass",
    var_name="measure",
    value_name="value"
)
display(class_summary_long)


# ==============================================================================
# 15. Flag possible outliers using the IQR rule
# ==============================================================================

q1 = df["fare"].quantile(0.25)
q3 = df["fare"].quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

df["fare_outlier_flag"] = (
    (df["fare"] < lower_fence) | (df["fare"] > upper_fence)
)

print(f"Q1: {q1:.2f}")
print(f"Q3: {q3:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Lower fence: {lower_fence:.2f}")
print(f"Upper fence: {upper_fence:.2f}")
print("Flagged fare observations:", int(df["fare_outlier_flag"].sum()))

display(df.loc[df["fare_outlier_flag"], [
    "passengerid", "pclass", "fare", "survived"
]].head(10))


# ==============================================================================
# PART 2 — DATA VISUALIZATION WITH MATPLOTLIB & PLOTLY
# ==============================================================================

plt.figure(figsize=(9, 5))
plt.hist(df["age"].dropna(), bins=20, edgecolor="black")
plt.title("Distribution of Passenger Ages")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# ==============================================================================
# 17. Matplotlib: bar chart of survival counts
# ==============================================================================

survival_counts_overall = df["survived"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(
    ["Did not survive", "Survived"],
    survival_counts_overall.values,
    edgecolor="black"
)
plt.title("Passenger Survival Counts")
plt.xlabel("Outcome")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()


# ==============================================================================
# 18. Matplotlib: survival rate by passenger class
# ==============================================================================

class_survival = df.groupby("pclass", observed=False)["survived"].mean() * 100

plt.figure(figsize=(7, 5))
plt.bar(class_survival.index.astype(str), class_survival.values, edgecolor="black")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()


# ==============================================================================
# 19. Matplotlib: box plot of fare by passenger class
# ==============================================================================

fare_groups = [
    df.loc[df["pclass"] == class_no, "fare"].dropna()
    for class_no in [1, 2, 3]
]

plt.figure(figsize=(8, 5))
plt.boxplot(fare_groups, labels=["First", "Second", "Third"])
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.show()


# ==============================================================================
# 20. Matplotlib: scatter plot of age and fare
# ==============================================================================

plt.figure(figsize=(8, 5))
plt.scatter(df["age"], df["fare"], alpha=0.55)
plt.title("Passenger Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.show()


# ==============================================================================
# 21. Matplotlib: correlation heatmap
# ==============================================================================

numeric_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare", "family_size"]
correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(9, 7))
plt.imshow(correlation_matrix, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Pearson correlation")
plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=45, ha="right")
plt.yticks(range(len(numeric_columns)), numeric_columns)
plt.title("Correlation Heatmap")
for i in range(len(numeric_columns)):
    for j in range(len(numeric_columns)):
        plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                 ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.show()


# ==============================================================================
# 22. Plotly: interactive survival rate by sex and class
# ==============================================================================

plot_df = (
    df.groupby(["sex", "pclass"], observed=False)["survived"]
      .mean()
      .reset_index()
)
plot_df["survival_percent"] = plot_df["survived"] * 100
plot_df["pclass"] = plot_df["pclass"].astype(str)

fig = px.bar(
    plot_df,
    x="sex",
    y="survival_percent",
    color="pclass",
    barmode="group",
    text="survival_percent",
    title="Survival Rate by Sex and Passenger Class",
    labels={
        "sex": "Sex",
        "survival_percent": "Survival rate (%)",
        "pclass": "Passenger class"
    }
)
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig.update_layout(yaxis_range=[0, 100])
fig.show()


# ==============================================================================
# 23. Plotly: interactive age distribution
# ==============================================================================

fig = px.histogram(
    df,
    x="age",
    color="survival_label",
    nbins=25,
    barmode="overlay",
    opacity=0.65,
    title="Age Distribution by Survival Outcome",
    labels={"age": "Age", "survival_label": "Survival outcome"}
)
fig.show()


# ==============================================================================
# 24. Plotly: family size and survival
# ==============================================================================

family_survival = (
    df.groupby("family_size", observed=False)["survived"]
      .agg(passengers="count", survival_rate="mean")
      .reset_index()
)
family_survival["survival_percent"] = family_survival["survival_rate"] * 100

fig = px.bar(
    family_survival,
    x="family_size",
    y="survival_percent",
    hover_data=["passengers"],
    title="Survival Rate by Family Size",
    labels={"family_size": "Family size", "survival_percent": "Survival rate (%)"}
)
fig.update_layout(yaxis_range=[0, 100])
fig.show()


# ==============================================================================
# PART 3 — STATISTICS WITH PYTHON: DESCRIPTIVE TO INFERENTIAL
# ==============================================================================

display(df[["age", "fare", "family_size"]].describe().T)

print("Mean age:", round(df["age"].mean(), 2))
print("Median age:", round(df["age"].median(), 2))
print("Age standard deviation:", round(df["age"].std(), 2))
print("Mean fare:", round(df["fare"].mean(), 2))
print("Median fare:", round(df["fare"].median(), 2))


# ==============================================================================
# 26. Descriptive statistics: categorical variables
# ==============================================================================

for column in ["sex", "pclass", "embarked", "survived", "is_alone"]:
    print(f"\\n{column.upper()} — count")
    display(df[column].value_counts(dropna=False).to_frame("count"))

    print(f"{column.upper()} — percentage")
    display((df[column].value_counts(normalize=True, dropna=False) * 100)
            .round(1).to_frame("percent"))


# ==============================================================================
# 27. Confidence interval for the overall survival proportion
# ==============================================================================

survived_count = int(df["survived"].sum())
total_count = int(df["survived"].count())
survival_proportion = survived_count / total_count

ci_low, ci_high = proportion_confint(
    count=survived_count,
    nobs=total_count,
    alpha=0.05,
    method="wilson"
)

print(f"Survived: {survived_count} out of {total_count}")
print(f"Observed survival proportion: {survival_proportion:.3f} ({survival_proportion*100:.1f}%)")
print(f"95% Wilson confidence interval: ({ci_low:.3f}, {ci_high:.3f})")
print(f"95% CI in percentage terms: ({ci_low*100:.1f}%, {ci_high*100:.1f}%)")


# ==============================================================================
# 28. Independent-samples Welch t-test: age by survival group
# ==============================================================================

age_survived = df.loc[df["survived"] == 1, "age"].dropna()
age_not_survived = df.loc[df["survived"] == 0, "age"].dropna()

t_stat, p_value = stats.ttest_ind(
    age_survived,
    age_not_survived,
    equal_var=False,
    alternative="two-sided"
)

mean_survived = age_survived.mean()
mean_not_survived = age_not_survived.mean()

print("Mean age — survived:", round(mean_survived, 2))
print("Mean age — did not survive:", round(mean_not_survived, 2))
print("Welch t statistic:", round(t_stat, 3))
print("p-value:", f"{p_value:.5f}")

alpha = 0.05
if p_value < alpha:
    print("Decision: Reject H0 at the 5% significance level.")
    print("Interpretation: Evidence of a difference in mean age between the two groups.")
else:
    print("Decision: Fail to reject H0 at the 5% significance level.")
    print("Interpretation: Insufficient evidence of a difference in mean age between the two groups.")


# ==============================================================================
# 29. Chi-square test of independence: sex and survival
# ==============================================================================

contingency_table = pd.crosstab(df["sex"], df["survived"])
display(contingency_table)

chi2, chi_p, dof, expected = stats.chi2_contingency(contingency_table)

expected_table = pd.DataFrame(
    expected,
    index=contingency_table.index,
    columns=contingency_table.columns
)

print("Expected counts under independence:")
display(expected_table.round(2))

print("Chi-square statistic:", round(chi2, 3))
print("Degrees of freedom:", dof)
print("p-value:", f"{chi_p:.8f}")

if (expected < 5).any():
    print("Caution: At least one expected cell count is below 5; review test assumptions.")

if chi_p < 0.05:
    print("Decision: Reject H0 at the 5% significance level.")
    print("Interpretation: Evidence of an association between sex and survival status.")
else:
    print("Decision: Fail to reject H0 at the 5% significance level.")
    print("Interpretation: Insufficient evidence of an association between sex and survival status.")


# ==============================================================================
# 30. Effect size for the chi-square test: Cramer's V
# ==============================================================================

n = contingency_table.to_numpy().sum()
r, k = contingency_table.shape
cramers_v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))

print("Cramer's V:", round(cramers_v, 3))
print("Cramer's V ranges from 0 (no association) to 1 (strongest possible association).")
print("Interpret its magnitude in the context of the table and subject area.")


# ==============================================================================
# 31. Correlation analysis
# ==============================================================================

correlation_matrix = df[numeric_columns].corr(method="pearson")
display(correlation_matrix.round(3))

print("Correlation between age and fare:",
      round(df["age"].corr(df["fare"]), 3))


# ==============================================================================
# PART 4 — FINAL DATA QUALITY CHECK & EXPORT
# ==============================================================================

print("Final shape:", df.shape)
print("\\nMissing values:")
display(df.isna().sum().to_frame("missing_count"))
print("Exact duplicate rows:", df.duplicated().sum())
print("\\nFinal data types:")
display(df.dtypes.to_frame("data_type"))


# ==============================================================================
# 33. Export cleaned data and figures
# ==============================================================================

cleaned_file = "titanic_cleaned_class03.csv"
df.to_csv(cleaned_file, index=False)
print("Saved:", cleaned_file)

# Save one example figure at 300 dpi.
plt.figure(figsize=(8, 5))
plt.bar(class_survival.index.astype(str), class_survival.values, edgecolor="black")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)
plt.tight_layout()
figure_file = "survival_by_class_class03.png"
plt.savefig(figure_file, dpi=300, bbox_inches="tight")
plt.show()

files.download(cleaned_file)
files.download(figure_file)


# ==============================================================================
# 34. Research interpretation: what should students report?
# ==============================================================================
