import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from scipy import stats
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

df = pd.read_csv("convertcsv (6).csv")
df = pd.DataFrame(df).set_index("Data")
print(df.shape)
print(df.describe())
print(df.info())
df = df.apply(pd.to_numeric, errors="coerce")
print(df.info())
print(df.describe())

df.plot(x="Minimalna temperatura powietrza", y="Maksymalna temperatura powietrza", style="X")
plt.title("Minimalna temperatura powietrza vs Maksymalna temperatura powietrza")
plt.xlabel("Minimalna temperatura powietrza")
plt.ylabel("Maksymalna temperatura powietrza")
# plt.xlim(1, 100)
plt.show()

plt.figure(figsize=(10, 5))
plt.tight_layout()
sns.displot(df["Maksymalna temperatura powietrza"])
plt.show()

X = df["Minimalna temperatura powietrza"].values.reshape(-1,1)
y = df["Maksymalna temperatura powietrza"].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm

print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)

df1 = pd.DataFrame({"Aktualna": y_test.flatten(), "Przewidziana": y_pred.flatten()})
print(df1)

# data splicing
df2 = df1.head(25)
df2.plot(kind="bar", figsize=(10, 5))
plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
plt.show()

plt.scatter(X_test, y_test, color="gray")
plt.plot(X_test, y_pred, color="red", linewidth=2)
plt.show()

print("Średni błąd bezwzględny: ", metrics.mean_absolute_error(y_test, y_pred))
print("Średni błąd kwadratowy: ", metrics.mean_squared_error(y_test, y_pred))
print("Podstawowy błąd średniokwadratowy: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
