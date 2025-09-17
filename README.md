# IDS706_mini_assignment_2
## Project Goal
This project aims to perform basic data analysis and machine learning on dataset gold_data_2015_25.csv to understand patterns in gold prices (GLD) and explore how other market indicators (SPX, USO, SLV, EUR/USD) relate to gold.

## Step 1
First create a new repository in github and clone it based on the following process (replace what's inside <> based on your url and your repository name):
```bash
git clone <repo_url>
cd <repo_name>
```
Then create a virtual environment:
```bash
python -m venv .venv
source .venv/Scripts/activate 
```
Install dependencies from requirements.txt with the following code:
```bash
pip install -r requirements.txt
```
Now you need to create a `.ipynb` file to work on the analysis.

## Step 2
1. Import & Inspect
Import modules we needed and import the dataset. Convert `Date` to datetime, and check the structure of the dataset with `.head()`, `.info()`, `.describe()`.
```bash
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("gold_data_2015_25.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

df.head()
df.info()
df.describe()
```

2. Filtering & Grouping
Filter GLD prices above the 75th percentile (high price periods) and group GLD by year to compute mean, std, min, max, and count.
```bash
gld_q75 = df["GLD"].quantile(0.75)
high_gld = df[df["GLD"] > gld_q75]
high_gld.head()

yearly_stats = df["GLD"].groupby(df.index.year).agg(["mean", "std", "min", "max", "count"])
yearly_stats
```
The filtering and grouping steps showed clear trends and volatility in gold prices, which implies SPX, USO, SLV, and EUR/USD may be good predictors for gold price. To test this assumption, let's do machine learning based on these variables.

## Step 3
3. Machine Learning and Visualization
Build a Linear Regression model to predict GLD from SPX, USO, SLV, and EUR/USD and evaluated using R² score and Mean Absolute Error (MAE). Here, we need to use sklearn to import `train_test_split`, `LinearRegression` and `mean_absolute_error`.
```bash
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

X = df[["SPX", "USO", "SLV", "EUR/USD"]]
y = df["GLD"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("Coefficients:", dict(zip(X.columns, model.coef_)))
```

Then plot True vs Predicted GLD to assess model performance.
```bash
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.title("True vs Predicted GLD")
plt.xlabel("True GLD")
plt.ylabel("Predicted GLD")
plt.show()
```

## Findings
After generating the final graph, here is what we will see:
1. The plot shows a strong positive correlation between true GLD values and the predictions from the linear regression model. Since most points are close to the diagonal, we can infer that SPX, USO, SLV, and EUR/USD contain useful information for predicting GLD.
2. However, we have to admit that there are some dispersion at higher GLD values, which imply that the model may not be useful in extreme prices.


