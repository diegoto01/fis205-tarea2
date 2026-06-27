import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from pathlib import Path


root = Path(__file__).resolve().parents[1]
fig_dir = root / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

K = 0.307075          # MeV cm^2 / mol
me = 0.51099895      # MeV
mp = 938.27208816    # MeV
rho = 1.0            # g/cm^3

z = 1
Z_A = 0.55509        # agua, mol/g
I = 75e-6            # MeV


def beta_gamma(E):
    gamma = 1 + E / mp
    beta2 = 1 - 1 / gamma**2
    beta = np.sqrt(beta2)
    return beta, gamma


def tmax(E):
    beta, gamma = beta_gamma(E)
    razon = me / mp

    num = 2 * me * beta**2 * gamma**2
    den = 1 + 2 * gamma * razon + razon**2

    return num / den


def stopping_power(E):
    E = np.asarray(E)
    E = np.maximum(E, 1e-6)

    beta, gamma = beta_gamma(E)
    beta2 = beta**2

    Tmax = tmax(E)

    arg = 2 * me * beta2 * gamma**2 * Tmax / I**2
    log_term = 0.5 * np.log(arg)

    S_masico = K * z**2 * Z_A * (1 / beta2) * (log_term - beta2)
    S = rho * S_masico

    return S


def rango_csda(E0):
    f = lambda E: 1 / stopping_power(E)
    R, err = quad(f, 1e-3, E0, limit=200)
    return R


energias = [50, 150, 250]

print("Rangos CSDA calculados")
print("----------------------")

for E0 in energias:
    R = rango_csda(E0)
    print(f"E0 = {E0:3.0f} MeV  ->  R = {R:.3f} cm  = {10*R:.1f} mm")


E_plot = np.linspace(1, 250, 600)
S_plot = stopping_power(E_plot)

plt.figure(figsize=(7, 5))
plt.plot(E_plot, S_plot)
plt.xlabel("Energía del protón [MeV]")
plt.ylabel("-dE/dx [MeV/cm]")
plt.title("Poder de frenado en agua")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "p2_stopping_power.png", dpi=300)
plt.close()

