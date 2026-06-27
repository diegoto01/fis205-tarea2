import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
fig_dir = root / "outputs" / "figures"

data = np.load(data_dir / "p1_dataset_sigma_0p02.npz")

t = data["t"]
X = data["X"]
y = data["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)


rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

pred_rf_train = rf.predict(X_train)
pred_rf_test = rf.predict(X_test)


scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    max_iter=300,
    random_state=42,
    early_stopping=True
)

mlp.fit(X_train_s, y_train)

pred_mlp_train = mlp.predict(X_train_s)
pred_mlp_test = mlp.predict(X_test_s)


def rmse_por_parametro(y_real, y_pred):
    rmse_gamma = np.sqrt(mean_squared_error(y_real[:, 0], y_pred[:, 0]))
    rmse_k = np.sqrt(mean_squared_error(y_real[:, 1], y_pred[:, 1]))
    return rmse_gamma, rmse_k


rf_train = rmse_por_parametro(y_train, pred_rf_train)
rf_test = rmse_por_parametro(y_test, pred_rf_test)

mlp_train = rmse_por_parametro(y_train, pred_mlp_train)
mlp_test = rmse_por_parametro(y_test, pred_mlp_test)

print("\nRandom Forest")
print("Train RMSE gamma, k:", rf_train)
print("Test  RMSE gamma, k:", rf_test)

print("\nMLP")
print("Train RMSE gamma, k:", mlp_train)
print("Test  RMSE gamma, k:", mlp_test)


plt.figure(figsize=(6, 5))
plt.scatter(y_test[:, 0], pred_rf_test[:, 0], s=12, alpha=0.6, label="gamma")
plt.scatter(y_test[:, 1], pred_rf_test[:, 1], s=12, alpha=0.6, label="k")

min_val = min(y_test.min(), pred_rf_test.min())
max_val = max(y_test.max(), pred_rf_test.max())

plt.plot([min_val, max_val], [min_val, max_val], "k--")
plt.xlabel("Valor real")
plt.ylabel("Valor predicho")
plt.title("Random Forest: predicción vs valor real")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "p1_rf_predicciones.png", dpi=300)
plt.show()


plt.figure(figsize=(6, 5))
plt.scatter(y_test[:, 0], pred_mlp_test[:, 0], s=12, alpha=0.6, label="gamma")
plt.scatter(y_test[:, 1], pred_mlp_test[:, 1], s=12, alpha=0.6, label="k")

min_val = min(y_test.min(), pred_mlp_test.min())
max_val = max(y_test.max(), pred_mlp_test.max())

plt.plot([min_val, max_val], [min_val, max_val], "k--")
plt.xlabel("Valor real")
plt.ylabel("Valor predicho")
plt.title("MLP: predicción vs valor real")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "p1_mlp_predicciones.png", dpi=300)
plt.show()
