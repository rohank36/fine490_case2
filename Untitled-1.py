# %%
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# # Data Cleaning

# %%
raw_data = pd.read_csv("data.csv")

# %%
raw_data

# %%
raw_data.info()

# %%
data = raw_data[["Company","Coattail","Sunset","Market Cap (C$ MM)","Subordinate Voting (MM)", "Superior Voting (MM)","Sup Vote/Econ","Stock Change 10-day %"]]

# %%
data

# %%
#data["Total Voting (MM)"] = data["Superior Voting (MM)"] + data["Subordinate Voting (MM)"]

# %%
#data.iloc[:,-1] = data.iloc[:,-1] /100

# %%
data.describe(include="all")

# %%
data["Coattail"] = data["Coattail"].map({"Yes": 1, "No": 0})
data["Sunset"]   = data["Sunset"].map({"Yes": 1, "No": 0})

# %%
MAGNA = data[data["Company"] == "Magna"].copy()

# %%
data = data.dropna().reset_index(drop=True)
#data = data.drop(12)

# %%
data.describe(include="all")

# %%
data.isnull().sum()     # count of missing values per column

# %%
data

# %% [markdown]
# # Training

# %%
X = data.iloc[:,1:-1]
Y = data.iloc[:,-1]

# %%
Y.plot(style="o", figsize=(6,4), title="% Change in Stock Price After 10 Days")
plt.show()

# Visually it's clear that regression is not a suitable learner for this data

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler

# %%
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# %%
model = LinearRegression()
model.fit(X_scaled,Y)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

y_pred = model.predict(X_scaled)
print("MSE:", mean_squared_error(Y, y_pred))
print("MAE :", mean_absolute_error(Y, y_pred))

results = pd.DataFrame({
    "Company": data["Company"].values,
    "y_true": Y.values ,   # actual
    "y_pred": y_pred    # predicted
})
results["error"] = results["y_true"] - results["y_pred"]
results["abs_error"] = results["error"].abs()
results



# %%
N_NEIGHBORS = 4

# %%
knn_model = KNeighborsRegressor(n_neighbors=N_NEIGHBORS)  
knn_model.fit(X_scaled, Y)
y_pred_knn = knn_model.predict(X_scaled)

print("MSE:", mean_squared_error(Y, y_pred_knn))
print("MAE:", mean_absolute_error(Y, y_pred_knn))

results_knn = pd.DataFrame({
    "Company": data["Company"].values,
    "y_true": Y.values,
    "y_pred": y_pred_knn
})
results_knn["error"] = results_knn["y_true"] - results_knn["y_pred"]
results_knn["abs_error"] = results_knn["error"].abs()
results_knn

# %%
N_ESTIMATORS = 10

# %%
rf_model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=42)
rf_model.fit(X, Y)
y_pred_rf = rf_model.predict(X)

print("MSE:", mean_squared_error(Y, y_pred_rf))
print("MAE:", mean_absolute_error(Y, y_pred_rf))

results_rf = pd.DataFrame({
    "Company": data["Company"].values,
    "y_true": Y.values,
    "y_pred": y_pred_rf
})
results_rf["error"] = results_rf["y_true"] - results_rf["y_pred"]
results_rf["abs_error"] = results_rf["error"].abs()
results_rf


# %%
importances = rf_model.feature_importances_
features = X.columns

feat_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(feat_importance_df)

# %%
MAX_DEPTH = 3

# %%
dt_model = DecisionTreeRegressor(random_state=42, max_depth=MAX_DEPTH)
dt_model.fit(X, Y)
y_pred_dt = dt_model.predict(X)

print("MSE:", mean_squared_error(Y, y_pred_dt))
print("MAE:", mean_absolute_error(Y, y_pred_dt))

results_dt = pd.DataFrame({
    "Company": data["Company"].values,
    "y_true": Y.values,
    "y_pred": y_pred_dt
})
results_dt["error"] = results_dt["y_true"] - results_dt["y_pred"]
results_dt["abs_error"] = results_dt["error"].abs()
results_dt

# %%

"""
plt.figure(figsize=(20,10)) 
tree.plot_tree(
    dt_model,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=9  
)
"""
#plt.savefig("dt.png", dpi=300, bbox_inches="tight") 
#plt.show()
#plt.close() 

# %%
loo = LeaveOneOut()

models = {
    "LinearRegression": Pipeline([
        ("model", LinearRegression())
    ]),
    "KNN": Pipeline([
        ("model", KNeighborsRegressor(n_neighbors=N_NEIGHBORS))
    ]),
    "RandomForest": RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=42),
    "DecisionTree": DecisionTreeRegressor(random_state=42,max_depth=MAX_DEPTH)
}


mae_scores = {}
preds_by_model = {}

for name, mdl in models.items():
    if mdl == "KNN" or mdl == "LinearRegression":
        y_pred = cross_val_predict(mdl, X_scaled, Y, cv=loo)
    else:
        y_pred = cross_val_predict(mdl, X, Y, cv=loo)
    preds_by_model[name] = y_pred
    mae = mean_absolute_error(Y, y_pred)
    mae_scores[name] = mae


print("LOOCV MAE by model:")
for name, mae in mae_scores.items():
    print(f"  {name:>14}: {mae:.4f}")

# (Optional) inspect per-point predictions for a model
# results = pd.DataFrame({"y_true": Y.values, "y_pred": preds_by_model["RandomForest"]})
# results["abs_err"] = (results["y_true"] - results["y_pred"]).abs()
# print(results)


# %% [markdown]
# # Inference

# %%
MAGNA

# %%
#MAGNA_X = MAGNA.iloc[:,1:-1]
MAGNA_X = MAGNA.loc[MAGNA["Company"]=="Magna", X.columns]
MAGNA_SCALED = pd.DataFrame(scaler.transform(MAGNA_X),columns=X.columns)

# %%
rf_pred = rf_model.predict(MAGNA_X)
knn_pred = knn_model.predict(MAGNA_SCALED)
reg_pred = model.predict(MAGNA_SCALED)
print(f"RF: {rf_pred}")
print(f"KNN: {knn_pred}")
print(f"Regression: {reg_pred}")

# %%
distances, indices = knn_model.kneighbors(MAGNA_SCALED, n_neighbors=len(X_scaled))

print("Neighbor ordering by distance:")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
    print(f"{rank}. {data.iloc[idx]['Company']}, Distance={dist:.3f}, 10 Day Stock Change %={Y.iloc[idx]}")

# %%
# 1. Pick Magna’s feature row (already aligned with X columns)
#magna_features = df.loc[df["Company"]=="Magna", X.columns].iloc[0]

# 2. Compute absolute differences for each company
abs_diffs = (X_scaled - MAGNA_SCALED).abs()

# 3. Plot for each company
fig, ax = plt.subplots(figsize=(12,6))

# Loop through companies (rows in abs_diffs)
for idx, row in abs_diffs.iterrows():
    ax.plot(row.index, row.values, marker="o", label=data.loc[idx, "Company"])

# 4. Formatting
ax.set_title("Absolute Feature Differences vs Magna")
ax.set_xlabel("Feature")
ax.set_ylabel("Absolute Difference (scaled units)")
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # legend outside chart
plt.tight_layout()
plt.show()


# %% [markdown]
# TODO: find way to plot or write in a table the distances for each feature between the training data and the MAGNA data

# %% [markdown]
# TODO: make sure you're doing everything correctly

# %% [markdown]
# KNN makes the most sense


