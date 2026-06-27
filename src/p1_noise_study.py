import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
fig_dir = root / "outputs" / "figures"

data_dir.mkdir(exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

N = 3000
Nt = 1000
t = np.linspace(0, 10, Nt)

sigmas = [0, 0.01, 0.02, 0.05, 0.10]


def edo(t, y, gamma, k):
    x, v = y
    return [v, -gamma * v - k * x]


def resolver(gamma, k):
    sol = solve_ivp(
        lambda t, y: edo(t, y, gamma, k),
        (t[0], t[-1]),
        [1, 0],
        t_eval=t,
        rtol=1e-8,
        atol=1e-10
    )
    return sol.y[0]


def crear_datos(sigma, seed=42):
    np.random.seed(seed)

    gammas = np.random.uniform(0.05, 1.0, N)
    ks = np.random.uniform(1.0, 5.0, N)

    X = np.zeros((N, Nt))
    y = np.zeros((N, 2))

    for i in range(N):
        x = resolver(gammas[i], ks[i])
        ruido = np.random.normal(0, sigma, Nt)

        X[i] = x + ruido
        y[i] = [gammas[i], ks[i]]

    return X, y


def rmse(y_real, y_pred):
    e_gamma = np.sqrt(mean_squared_error(y_real[:, 0], y_pred[:, 0]))
    e_k = np.sqrt(mean_squared_error(y_real[:, 1], y_pred[:, 1]))
    return e_gamma, e_k


res = {
    "sigma": [],
    "rf_gamma_train": [],
    "rf_k_train": [],
    "rf_gamma_test": [],
    "rf_k_test": [],
    "mlp_gamma_train": [],
    "mlp_k_train": [],
    "mlp_gamma_test": [],
    "mlp_k_test": []
}


for sigma in sigmas:
    print("\nsigma =", sigma)

    X, y = crear_datos(sigma)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=100,
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

    rf_train = rmse(y_train, pred_rf_train)
    rf_test = rmse(y_test, pred_rf_test)

    mlp_train = rmse(y_train, pred_mlp_train)
    mlp_test = rmse(y_test, pred_mlp_test)

    res["sigma"].append(sigma)

    res["rf_gamma_train"].append(rf_train[0])
    res["rf_k_train"].append(rf_train[1])
    res["rf_gamma_test"].append(rf_test[0])
    res["rf_k_test"].append(rf_test[1])

    res["mlp_gamma_train"].append(mlp_train[0])
    res["mlp_k_train"].append(mlp_train[1])
    res["mlp_gamma_test"].append(mlp_test[0])
    res["mlp_k_test"].append(mlp_test[1])

    print("RF train:", rf_train)
    print("RF test :", rf_test)
    print("MLP train:", mlp_train)
    print("MLP test :", mlp_test)


np.savez(data_dir / "p1_noise_results.npz", **res)

s = np.array(res["sigma"])

plt.figure(figsize=(7, 5))
plt.plot(s, res["rf_gamma_test"], "o-", label="RF test")
plt.plot(s, res["mlp_gamma_test"], "o-", label="MLP test")
plt.xlabel("sigma")
plt.ylabel("RMSE gamma")
plt.title("Error en gamma al variar el ruido")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "p1_ruido_gamma.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
plt.plot(s, res["rf_k_test"], "o-", label="RF test")
plt.plot(s, res["mlp_k_test"], "o-", label="MLP test")
plt.xlabel("sigma")
plt.ylabel("RMSE k")
plt.title("Error en k al variar el ruido")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "p1_ruido_k.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
plt.plot(s, res["rf_gamma_train"], "o--", label="RF gamma train")
plt.plot(s, res["rf_gamma_test"], "o-", label="RF gamma test")
plt.plot(s, res["rf_k_train"], "o--", label="RF k train")
plt.plot(s, res["rf_k_test"], "o-", label="RF k test")
plt.xlabel("sigma")
plt.ylabel("RMSE")
plt.title("Random Forest: train vs test")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "p1_rf_train_test.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
plt.plot(s, res["mlp_gamma_train"], "o--", label="MLP gamma train")
plt.plot(s, res["mlp_gamma_test"], "o-", label="MLP gamma test")
plt.plot(s, res["mlp_k_train"], "o--", label="MLP k train")
plt.plot(s, res["mlp_k_test"], "o-", label="MLP k test")
plt.xlabel("sigma")
plt.ylabel("RMSE")
plt.title("MLP: train vs test")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "p1_mlp_train_test.png", dpi=300)
plt.close()
