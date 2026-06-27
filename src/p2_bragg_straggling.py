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

re = 2.8179403262e-13
NA = 6.02214076e23
Ne = rho * Z_A * NA

E0 = 150.0
N = 10000
dx = 0.01
zmax = 25.0

np.random.seed(42)


def beta_gamma(E):
    gamma = 1 + E / mp
    beta2 = 1 - 1 / gamma**2
    beta2 = np.maximum(beta2, 1e-12)
    return np.sqrt(beta2), gamma


def tmax(E):
    beta, gamma = beta_gamma(E)
    r = me / mp
    return 2 * me * beta**2 * gamma**2 / (1 + 2 * gamma * r + r**2)


def stopping_power(E):
    E = np.asarray(E)
    E = np.maximum(E, 1e-6)

    beta, gamma = beta_gamma(E)
    beta2 = beta**2
    Tmax = tmax(E)

    arg = 2 * me * beta2 * gamma**2 * Tmax / I**2
    arg = np.maximum(arg, 1.0000001)

    log_term = 0.5 * np.log(arg)
    S = K * zp**2 * Z_A * (1 / beta2) * (log_term - beta2)

    return rho * S


def sigma_energy(E):
    E = np.asarray(E)
    E = np.maximum(E, 1e-6)

    beta, gamma = beta_gamma(E)
    beta2 = beta**2

    var = 4 * np.pi * re**2 * me**2 * Ne * zp**2 * dx / beta2
    var = np.maximum(var, 0)

    return np.sqrt(var)


def curva_determinista():
    bins = np.arange(0, zmax + dx, dx)
    dose = np.zeros(len(bins) - 1)

    E = E0
    z = 0.0

    while E > 0 and z < zmax:
        dE = stopping_power(E) * dx
        dE = min(dE, E)

        i = int(z / dx)
        dose[i] += dE * N

        E -= dE
        z += dx

    prof = 0.5 * (bins[:-1] + bins[1:])
    return prof, dose


def curva_con_straggling():
    bins = np.arange(0, zmax + dx, dx)
    dose = np.zeros(len(bins) - 1)

    E = np.full(N, E0)
    alive = np.ones(N, dtype=bool)
    ranges = np.full(N, np.nan)

    nsteps = len(bins) - 1

    for i in range(nsteps):
        if not np.any(alive):
            break

        idx = np.where(alive)[0]
        Ea = E[idx]

        mean_dE = stopping_power(Ea) * dx
        sig_dE = sigma_energy(Ea)

        fluct = np.random.normal(0, sig_dE)
        fluct = np.clip(fluct, -0.8 * mean_dE, 0.8 * mean_dE)

        dE = mean_dE + fluct
        dE = np.minimum(dE, Ea)

        dose[i] += np.sum(dE)
        E[idx] -= dE

        stopped = idx[E[idx] <= 1e-6]
        ranges[stopped] = (i + 1) * dx
        alive[stopped] = False

    ranges[np.isnan(ranges)] = zmax

    prof = 0.5 * (bins[:-1] + bins[1:])
    return prof, dose, ranges


prof_det, dose_det = curva_determinista()
prof_str, dose_str, ranges = curva_con_straggling()

pico_det = prof_det[np.argmax(dose_det)]
pico_str = prof_str[np.argmax(dose_str)]
R_mean = np.mean(ranges)
sigma_R = np.std(ranges)

print(f"Pico determinista: {pico_det:.3f} cm")
print(f"Pico con straggling: {pico_str:.3f} cm")
print(f"Rango medio: {R_mean:.3f} cm")
print(f"sigma_R: {sigma_R:.3f} cm")

plt.figure(figsize=(8, 5))
plt.plot(prof_det, dose_det, label="Sin straggling")
plt.plot(prof_str, dose_str, label="Con straggling")
plt.axvline(pico_det, linestyle="--", label=f"Pico det = {pico_det:.2f} cm")
plt.axvline(pico_str, linestyle=":", label=f"Pico str = {pico_str:.2f} cm")
plt.xlabel("Profundidad z [cm]")
plt.ylabel("Energía depositada [MeV]")
plt.title("Pico de Bragg con y sin straggling")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "p2_bragg_comparacion.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.hist(ranges, bins=50)
plt.xlabel("Rango final [cm]")
plt.ylabel("Número de protones")
plt.title("Distribución de rangos finales")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "p2_rangos_hist.png", dpi=300)
plt.close()
