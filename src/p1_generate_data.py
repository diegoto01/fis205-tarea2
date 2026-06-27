import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

np.random.seed(42)

N = 3000
Nt = 1000
sigma = 0.02

t = np.linspace(0, 10, Nt)

root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
fig_dir = root / "outputs" / "figures"

data_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)


def edo(t, y, gamma, k):
    x, v = y
    return [v, -gamma * v - k * x]


def resolver_oscilador(gamma, k):
    sol = solve_ivp(
        lambda t, y: edo(t, y, gamma, k),
        (t[0], t[-1]),
        [1, 0],
        t_eval=t,
        rtol=1e-8,
        atol=1e-10
    )
    return sol.y[0]


gammas = np.random.uniform(0.05, 1.0, N)
ks = np.random.uniform(1.0, 5.0, N)

X = np.zeros((N, Nt))
y = np.zeros((N, 2))

for i in range(N):
    x = resolver_oscilador(gammas[i], ks[i])
    ruido = np.random.normal(0, sigma, Nt)

    X[i] = x + ruido
    y[i] = [gammas[i], ks[i]]

    if (i + 1) % 500 == 0:
        print(i + 1, "señales generadas")

np.savez_compressed(data_dir / "p1_dataset_sigma_0p02.npz", t=t, X=X, y=y)

plt.figure(figsize=(9, 5))

for i in range(6):
    plt.plot(t, X[i], label=fr"$\gamma={y[i,0]:.2f}$, $k={y[i,1]:.2f}$")

plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Señales simuladas con ruido")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "p1_senales.png", dpi=300)
plt.show()
