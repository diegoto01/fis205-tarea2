import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


root = Path(__file__).resolve().parents[1]
fig_dir = root / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

K = 0.307075
me = 0.51099895
mp = 938.27208816
rho = 1.0
Z_A = 0.55509
I = 75e-6
zp = 1

E0 = 150.0
N = 10000
dx_cm = 0.01        # 0.1 mm = 0.01 cm
zmax = 25.0


def beta_gamma(E):
    gamma = 1 + E / mp
    beta2 = 1 - 1 / gamma**2
    return np.sqrt(beta2), gamma


def tmax(E):
    beta, gamma = beta_gamma(E)
    r = me / mp
    return 2 * me * beta**2 * gamma**2 / (1 + 2 * gamma * r + r**2)


def stopping_power(E):
    E = max(E, 1e-6)

    beta, gamma = beta_gamma(E)
    beta2 = beta**2
    Tmax = tmax(E)

    arg = 2 * me * beta2 * gamma**2 * Tmax / I**2
    log_term = 0.5 * np.log(arg)

    S = K * zp**2 * Z_A * (1 / beta2) * (log_term - beta2)
    return rho * S


bins = np.arange(0, zmax + dx_cm, dx_cm)
dose = np.zeros(len(bins) - 1)

E = E0
z = 0.0

while E > 0 and z < zmax:
    S = stopping_power(E)
    dE = S * dx_cm

    if dE > E:
        dE = E

    i = int(z / dx_cm)
    dose[i] += dE * N

    E -= dE
    z += dx_cm


prof = 0.5 * (bins[:-1] + bins[1:])
pico = prof[np.argmax(dose)]

print(f"Rango simulado: {z:.3f} cm")
print(f"Pico de Bragg: {pico:.3f} cm")

plt.figure(figsize=(8, 5))
plt.plot(prof, dose)
plt.axvline(pico, linestyle="--", label=f"Pico = {pico:.2f} cm")
plt.xlabel("Profundidad z [cm]")
plt.ylabel("Energía depositada [MeV]")
plt.title("Pico de Bragg para protones de 150 MeV en agua")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "p2_bragg_determinista.png", dpi=300)
plt.close()
