!pip install skimpy 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import missingno as msno
from skimpy import skim
from scipy.stats import skew

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from yellowbrick.regressor import ResidualsPlot, PredictionError
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet

import warnings
warnings.filterwarnings('ignore')

# Load dataset
data_path = "CO2 Emissions_Canada.csv"
df = pd.read_csv(data_path)

# Output heading
print("=== Dataset Head ===")
print(df.head())
print()
print()

# Preprocessing 
df.columns = df.columns.str.replace(" (L/100 km)", "").str.replace("(L)", "").str.replace("(g/km)", "").str.replace(" ", "_")
df.columns = df.columns.str.lower()
df = df.drop_duplicates(keep='first').reset_index(drop=True)
print()
print()

# Output heading
print("=== Descriptive Statistics (Numerical) ===")
print(df.describe().T)
print()
print()

print("=== Descriptive Statistics (Categorical) ===")
print(df.describe(include="object").T)
print()
print()

# Missing values
print("=== Missing Value Visualization ===")
msno.bar(df, color='green')
plt.show()
print()
print()

print("=== Pairplot for Numeric Columns ===")
sns.pairplot(df)
plt.show()
print()
print()

# Correlation matrix
print("=== Correlation Matrix ===")
numeric_df = df.select_dtypes(exclude="object")
plt.figure(figsize=(9, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu")
plt.title('Correlation Matrix', fontsize=18)
plt.show()
print()
print()

# Vehicle class distribution
print("=== Vehicle Class Distribution ===")
plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='vehicle_class', palette='summer')
plt.title('Vehicle Class Distribution')
plt.show()
print()
print()

# CO2 emissions by vehicle class
print("=== CO2 Emissions by Vehicle Class ===")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="vehicle_class", y="co2_emissions", palette="YlGn")
plt.title("CO2 Emissions by Vehicle Class")
plt.xticks(rotation=30)
plt.show()
print()
print()

# Average fuel consumption
print("=== Average Fuel Consumption by Vehicle Class ===")
avg_fuel = df.groupby('vehicle_class')['fuel_consumption_comb'].mean().sort_values()
avg_fuel.plot(kind='bar', cmap='summer')
plt.title('Average Fuel Consumption by Vehicle Class')
plt.ylabel('Fuel Consumption (Combined)')
plt.show()
print()
print()

# FacetGrid plot
print("=== Engine Size vs CO2 Emissions by Fuel Type ===")
g = sns.FacetGrid(df, col='fuel_type', height=5, aspect=1)
g.map(sns.scatterplot, 'engine_size', 'co2_emissions', color='green')
g.add_legend()
plt.show()

# Machine Learning Model
X = df[["engine_size"]]
y = df["co2_emissions"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print()
print()

# Output heading
print("=== Linear Regression Coefficients ===")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
print()

# Evaluation function
def train_val(y_train, y_train_pred, y_test, y_pred, i):
    scores = {
        i+"_train": {
            "R2" : r2_score(y_train, y_train_pred),
            "mae" : mean_absolute_error(y_train, y_train_pred),
            "mse" : mean_squared_error(y_train, y_train_pred),
            "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))
        },
        i+"_test": {
            "R2" : r2_score(y_test, y_pred),
            "mae" : mean_absolute_error(y_test, y_pred),
            "mse" : mean_squared_error(y_test, y_pred),
            "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))
        }
    }
    return pd.DataFrame(scores)

# Evaluate simple linear model
y_train_pred = model.predict(X_train)
slr = train_val(y_train, y_train_pred, y_test, y_pred, 'linear')
print()
print()

# Output heading
print("=== Simple Linear Regression Performance ===")
print(slr)
print()

avg_co2 = df['co2_emissions'].mean()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
accuracy_rate = ((1 - rmse / avg_co2) * 100).round(1)
print()
print()

print("=== RMSE and Accuracy ===")
print(f"RMSE: {rmse}")
print(f"Accuracy rate: {accuracy_rate}%")
print()
print()

# Residuals plot
print("=== Residuals Plot ===")
visualizer = ResidualsPlot(model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
print()
print()

# Prediction error plot
print("=== Prediction Error Plot ===")
visualizer = PredictionError(model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# Final model with multiple features
X = df[["engine_size", "fuel_consumption_city", "fuel_consumption_hwy", "fuel_consumption_comb"]]
y = df["co2_emissions"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)
mlr = train_val(y_train, y_train_pred, y_test, y_pred, "linear")
print()
print()

print("=== Multiple Linear Regression Performance ===")
print(mlr)
print()
print()

# Cross-validation
print("=== Cross Validation Scores ===")
scores = cross_validate(estimator=model, X=X_train, y=y_train,
                        scoring=["r2", "neg_mean_absolute_error", "neg_mean_squared_error", "neg_root_mean_squared_error"],
                        cv=10, return_train_score=True)
scores_df = pd.DataFrame(scores, index=range(1, 11))
print(scores_df.iloc[:, 2:].mean())
print()
print()

# Ridge Regression
print("=== Ridge Regression Grid Search ===")
alpha_space = np.linspace(0.01, 1, 100)
param_grid = {"alpha": alpha_space}

grid_ridge = GridSearchCV(estimator=Ridge(),
                          param_grid=param_grid,
                          scoring="neg_root_mean_squared_error",
                          cv=10, verbose=1, return_train_score=True)
grid_ridge.fit(X_train, y_train)

print("Best Ridge Alpha:", grid_ridge.best_params_)
print()

y_pred = grid_ridge.predict(X_test)
y_train_pred = grid_ridge.predict(X_train)
rm = Ridge(alpha=1.0).fit(X_train, y_train)
rm = train_val(y_train, y_train_pred, y_test, y_pred, "ridge")
print()
print()

print("=== Ridge Regression Performance ===")
print(rm)
print()
print()
print()

# Lasso Regression
print("=== Lasso Regression Grid Search ===")
grid_lasso = GridSearchCV(estimator=Lasso(),
                          param_grid=param_grid,
                          scoring="neg_root_mean_squared_error",
                          cv=10, verbose=1, return_train_score=True)
grid_lasso.fit(X_train, y_train)

print("Best Lasso Alpha:", grid_lasso.best_params_)
print()

y_pred = grid_lasso.predict(X_test)
y_train_pred = grid_lasso.predict(X_train)
lss = Lasso(alpha=0.01).fit(X_train, y_train)
lss = train_val(y_train, y_train_pred, y_test, y_pred, "lasso")
print()
print()

print("=== Lasso Regression Performance ===")
print(lss)
print()
print()
print()

# Final prediction
print("=== Final CO2 Emission Prediction ===")
final_model = ElasticNet(alpha=0.01, l1_ratio=0.1).fit(X, y)
new_data = [[2.0, 11.6, 7.2, 10.4]]
CO2 = final_model.predict(new_data)
CO2 = np.squeeze(CO2).round(1)

print(f"The CO2 emissions of this vehicle will be around {CO2} g/Km with an accuracy rate of {accuracy_rate}%")
print()