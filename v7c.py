import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# KF v7c : mass-potential-support + lensing criticality observer
#
# State:
#   rho(r,t)   : density / mass field
#   psi(r,t)   : closure / potential field
#   sigma(r,t) : support / dispersion field
#
# Goal:
#   test whether lensing-like ring criticality comes from:
#   (A) shell-assisted pile-up
#   (B) smooth projected criticality
#
# Notes:
#   - dynamics and lensing geometry are separated
#   - Sigma_crit is an observational geometry parameter, not dynamics
# ============================================================

OUTDIR = Path("kf_v7c_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 7
rng = np.random.default_rng(SEED)

# ----------------------------
# radial grid
# ----------------------------
N = 360
RMAX = 20.0
dr = RMAX / N
r = (np.arange(N) + 0.5) * dr

# ----------------------------
# runtime
# ----------------------------
steps = 10000
dt = 2.0e-4
diag_every = 500

# ----------------------------
# dynamics parameters
# ----------------------------
G = 1.0
soft = 0.35

D_rho = 0.018
D_psi = 0.080
D_sigma = 0.035

chi = 0.58          # inward gravity-drift strength
kappa_p = 0.72      # support / pressure strength

lambda_Q = 2.6      # psi -> psi_star relaxation
lambda_sigma = 1.8  # sigma -> sigma_eq relaxation

beta_sat = 0.10     # anti-blowup saturation

sigma_floor = 0.025
a_rho = 0.85
rho0 = 0.18
a_g = 0.72
g0 = 0.18
a_B = 0.28

a_comp = 0.75
comp0 = 0.05

# structured infall eta
A_eta = 0.014
r_eta = 13.8
w_eta = 1.15

# weak outer sink to prevent boundary pile-up
outer_sink_amp = 0.014
outer_sink_center = 17.6
outer_sink_width = 0.75

# ----------------------------
# lensing geometry parameters
# not dynamics
# ----------------------------
SIGMA_CRIT_LIST = [0.35, 0.55, 0.80]
SIGMA_CRIT_MAIN = 0.55
CRIT_THRESH = 0.08   # |lambda_t| below this => near-critical

# ----------------------------
# helper functions
# ----------------------------
EPS = 1e-12

def grad(f, dr):
    g = np.empty_like(f)
    g[1:-1] = (f[2:] - f[:-2]) / (2.0 * dr)
    g[0] = (f[1] - f[0]) / dr
    g[-1] = (f[-1] - f[-2]) / dr
    return g

def div_flux_sph(J, r, dr):
    h = (r**2) * J
    dh = grad(h, dr)
    return dh / np.maximum(r**2, 1e-12)

def lap_sph(f, r, dr):
    return div_flux_sph(grad(f, dr), r, dr)

def enclosed_mass(r, rho, dr):
    # M(r) = 4 pi ∫ rho r^2 dr
    return 4.0 * np.pi * np.cumsum(rho * r**2) * dr

def fit_log_slope(x, y, xmin, xmax, floor=1e-12):
    m = (x > xmin) & (x < xmax) & (y > floor)
    if np.count_nonzero(m) < 8:
        return np.nan
    a, _ = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    return a

def abel_projection(r, rho):
    # Sigma(R) = 2 ∫_R^∞ rho(r) r / sqrt(r^2 - R^2) dr
    R = r.copy()
    Sigma = np.zeros_like(R)
    for i, Ri in enumerate(R):
        mask = r > Ri
        rr = r[mask]
        rh = rho[mask]
        if rr.size < 2:
            Sigma[i] = 0.0
            continue
        kern = 2.0 * rh * rr / np.sqrt(np.maximum(rr**2 - Ri**2, 1e-12))
        Sigma[i] = np.trapz(kern, rr)
    return R, Sigma

def mean_cylindrical(R, f):
    # \bar f(<R) = (2/R^2) ∫_0^R f(R') R' dR'
    out = np.zeros_like(f)
    acc = np.cumsum(f * R) * (R[1] - R[0])
    out[1:] = 2.0 * acc[1:] / np.maximum(R[1:]**2, 1e-12)
    out[0] = f[0]
    return out

def find_balance_radius(R, J, rmin=2.0, rmax=16.0):
    m = (R >= rmin) & (R <= rmax)
    RR = R[m]
    JJ = J[m]
    if RR.size < 3:
        return np.nan
    s = np.sign(JJ)
    idx = np.where(s[:-1] * s[1:] <= 0)[0]
    if idx.size > 0:
        return RR[idx[0]]
    return RR[np.argmin(np.abs(JJ))]

def find_outer_shell_radius(R, rho, rmin=6.0, rmax=16.0):
    m = (R >= rmin) & (R <= rmax)
    if np.count_nonzero(m) < 3:
        return np.nan
    RR = R[m]
    YY = rho[m]
    i = np.argmax(YY)
    if i == 0 or i == len(YY) - 1:
        return np.nan
    return RR[i]

def shell_ratio(R, rho):
    inner = (R > 0.8) & (R < 4.0)
    outer = (R > 7.0) & (R < 15.0)
    if not np.any(inner) or not np.any(outer):
        return np.nan
    inner_peak = np.max(rho[inner])
    outer_peak = np.max(rho[outer])
    return outer_peak / max(inner_peak, 1e-12)

def lensing_diagnostics(R, Sigma, sigma_crit):
    kappa = Sigma / max(sigma_crit, 1e-12)
    bar_kappa = mean_cylindrical(R, kappa)
    gamma_t = bar_kappa - kappa
    lambda_t = 1.0 - kappa - gamma_t      # = 1 - bar_kappa
    lambda_r = 1.0 - kappa + gamma_t      # = 1 + bar_kappa - 2kappa

    i = np.argmin(np.abs(lambda_t))
    Rcrit = R[i]
    crit_strength = np.abs(lambda_t[i])

    if crit_strength > CRIT_THRESH:
        Rcrit = np.nan

    return {
        "kappa": kappa,
        "bar_kappa": bar_kappa,
        "gamma_t": gamma_t,
        "lambda_t": lambda_t,
        "lambda_r": lambda_r,
        "Rcrit": Rcrit,
        "crit_strength": crit_strength,
    }

# ----------------------------
# initial conditions
# cored center + weak outer bump
# deliberately lets us test shell-vs-criticality
# ----------------------------
rho = (
    0.75 * np.exp(-(r / 2.15)**2)
    + 0.030 * np.exp(-0.5 * ((r - 11.6) / 1.0)**2)
)
rho *= (1.0 + 0.03 * rng.standard_normal(N))
rho = np.clip(rho, 1e-8, None)

# weak background scaffold
B = (1.0 / (1.0 + (r / 9.0)**2)) * np.exp(-(r / 18.0)**8)

# closure initial
M = enclosed_mass(r, rho, dr)
g_star = G * M / (r**2 + soft**2)
psi = np.cumsum(g_star) * dr

# support initial
sigma = sigma_floor + 0.06 * np.exp(-(r / 3.0)**2)

# ----------------------------
# history
# ----------------------------
hist_step = []
hist_slope_rho = []
hist_slope_M = []
hist_slope_Sigma = []
hist_flatness_v = []
hist_shell = []
hist_balance = []
hist_crit_main = []

# ----------------------------
# main simulation
# ----------------------------
for step in range(1, steps + 1):
    # ----- closure target from current rho -----
    M = enclosed_mass(r, rho, dr)
    g_target = G * M / (r**2 + soft**2)
    psi_star = np.cumsum(g_target) * dr

    # ----- update psi -----
    dpsi = D_psi * lap_sph(psi, r, dr) - lambda_Q * (psi - psi_star)
    psi = psi + dt * dpsi

    g = np.maximum(grad(psi, dr), 0.0)

    # ----- compression proxy from gravity flux -----
    J_grav = -chi * rho * g
    comp = np.maximum(-div_flux_sph(J_grav, r, dr), 0.0)
    comp_drive = comp / (comp + comp0)

    # ----- support equilibrium -----
    sigma_eq = (
        sigma_floor
        + a_rho * (rho / (rho + rho0))
        + a_g * (g / (g + g0))
        + a_B * B
    )

    dsigma = (
        D_sigma * lap_sph(sigma, r, dr)
        + lambda_sigma * (sigma_eq - sigma)
        + a_comp * comp_drive
    )
    sigma = sigma + dt * dsigma
    sigma = np.clip(sigma, sigma_floor, None)

    # ----- density flux -----
    pressure_term = sigma * rho
    J = (
        -D_rho * grad(rho, dr)
        -chi * rho * g
        -kappa_p * grad(pressure_term, dr)
    )

    # structured infall
    S_eta = A_eta * np.exp(-0.5 * ((r - r_eta) / w_eta)**2)

    # weak outer sink
    outer_window = 1.0 / (1.0 + np.exp(-(r - outer_sink_center) / outer_sink_width))
    S_sink = outer_sink_amp * rho * outer_window

    drho = (
        -div_flux_sph(J, r, dr)
        + S_eta
        - beta_sat * rho**2
        - S_sink
    )

    rho = rho + dt * drho
    rho = np.clip(rho, 1e-8, None)

    # ----- diagnostics -----
    if step % diag_every == 0 or step == 1:
        M = enclosed_mass(r, rho, dr)
        v = np.sqrt(np.maximum(G * M / np.maximum(r, 1e-8), 1e-12))

        Rproj, Sigma = abel_projection(r, rho)
        lens_main = lensing_diagnostics(Rproj, Sigma, SIGMA_CRIT_MAIN)

        slope_rho = fit_log_slope(r, rho, 3.5, 13.5)
        slope_M = fit_log_slope(r, M, 3.5, 13.5)
        slope_Sigma = fit_log_slope(Rproj, Sigma, 3.5, 13.5)

        flat_mask = (r > 4.0) & (r < 13.5)
        flatness_v = np.std(v[flat_mask]) / np.mean(v[flat_mask])

        sr = shell_ratio(r, rho)
        rb = find_balance_radius(r, J, 2.0, 16.0)

        hist_step.append(step)
        hist_slope_rho.append(slope_rho)
        hist_slope_M.append(slope_M)
        hist_slope_Sigma.append(slope_Sigma)
        hist_flatness_v.append(flatness_v)
        hist_shell.append(sr)
        hist_balance.append(rb)
        hist_crit_main.append(lens_main["Rcrit"])

        print(
            f"[step {step:5d}] "
            f"slope_rho ~ {slope_rho:+.4f}   "
            f"slope_M ~ {slope_M:+.4f}   "
            f"slope_Sigma ~ {slope_Sigma:+.4f}   "
            f"flatness(v) ~ {flatness_v:.4f}   "
            f"shell_ratio ~ {sr:.4f}   "
            f"R_balance ~ {rb:.3f}   "
            f"Rcrit(main) ~ {lens_main['Rcrit']}"
        )

# ----------------------------
# final diagnostics
# ----------------------------
M = enclosed_mass(r, rho, dr)
v = np.sqrt(np.maximum(G * M / np.maximum(r, 1e-8), 1e-12))
Rproj, Sigma = abel_projection(r, rho)

slope_rho = fit_log_slope(r, rho, 3.5, 13.5)
slope_M = fit_log_slope(r, M, 3.5, 13.5)
slope_Sigma = fit_log_slope(Rproj, Sigma, 3.5, 13.5)
flat_mask = (r > 4.0) & (r < 13.5)
flatness_v = np.std(v[flat_mask]) / np.mean(v[flat_mask])

sr = shell_ratio(r, rho)
R_balance = find_balance_radius(r, J, 2.0, 16.0)
R_shell = find_outer_shell_radius(r, rho, 6.0, 16.0)

lens_by_geom = {}
for sc in SIGMA_CRIT_LIST:
    lens_by_geom[sc] = lensing_diagnostics(Rproj, Sigma, sc)

# heuristic interpretation
Rcrit_main = lens_by_geom[SIGMA_CRIT_MAIN]["Rcrit"]
if np.isnan(Rcrit_main):
    ring_mode = "subcritical / no tangential critical curve"
elif np.isnan(R_shell):
    ring_mode = "smooth-criticality favored"
elif abs(Rcrit_main - R_shell) <= 1.5:
    ring_mode = "shell-assisted criticality favored"
else:
    ring_mode = "smooth-criticality favored"

print("\n==== KF v7c summary ====")
print(f"density slope fit         ~ {slope_rho:+.4f}")
print(f"mass slope fit            ~ {slope_M:+.4f}")
print(f"projected slope fit       ~ {slope_Sigma:+.4f}")
print(f"v flatness(std/mean)      ~ {flatness_v:.4f}")
print(f"shell_ratio               ~ {sr:.4f}")
print(f"R_balance                 ~ {R_balance}")
print(f"R_shell                   ~ {R_shell}")
print(f"main geometry Sigma_crit  = {SIGMA_CRIT_MAIN:.4f}")
print(f"main geometry Rcrit       ~ {Rcrit_main}")
print(f"mode guess                = {ring_mode}")

for sc in SIGMA_CRIT_LIST:
    ld = lens_by_geom[sc]
    print(
        f"Sigma_crit={sc:.3f}  "
        f"Rcrit={ld['Rcrit']}  "
        f"min|lambda_t|={ld['crit_strength']:.4f}"
    )

# save arrays
save_dict = {
    "r": r,
    "rho": rho,
    "psi": psi,
    "sigma": sigma,
    "M": M,
    "v": v,
    "Rproj": Rproj,
    "Sigma": Sigma,
    "slope_rho": np.array(hist_slope_rho),
    "slope_M": np.array(hist_slope_M),
    "slope_Sigma": np.array(hist_slope_Sigma),
    "flatness_v": np.array(hist_flatness_v),
    "shell_ratio_hist": np.array(hist_shell),
    "balance_hist": np.array(hist_balance),
    "crit_main_hist": np.array(hist_crit_main),
    "hist_step": np.array(hist_step),
    "R_balance_final": R_balance,
    "R_shell_final": R_shell,
}
for sc in SIGMA_CRIT_LIST:
    ld = lens_by_geom[sc]
    save_dict[f"kappa_{sc:.3f}"] = ld["kappa"]
    save_dict[f"bar_kappa_{sc:.3f}"] = ld["bar_kappa"]
    save_dict[f"gamma_t_{sc:.3f}"] = ld["gamma_t"]
    save_dict[f"lambda_t_{sc:.3f}"] = ld["lambda_t"]
    save_dict[f"lambda_r_{sc:.3f}"] = ld["lambda_r"]

np.savez(OUTDIR / "kf_v7c_final.npz", **save_dict)

# ----------------------------
# plots
# ----------------------------
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# 1) rho and sigma
axs[0, 0].plot(r, rho, label="rho")
axs[0, 0].plot(r, sigma, label="sigma")
if not np.isnan(R_shell):
    axs[0, 0].axvline(R_shell, linestyle="--", label="R_shell")
if not np.isnan(R_balance):
    axs[0, 0].axvline(R_balance, linestyle=":", label="R_balance")
axs[0, 0].set_title("rho(r), sigma(r)")
axs[0, 0].set_xlabel("r")
axs[0, 0].legend()
axs[0, 0].grid(alpha=0.25)

# 2) M and v
axs[0, 1].plot(r, M, label="M(r)")
axs[0, 1].plot(r, v, label="v(r)")
axs[0, 1].set_title("mass and rotation")
axs[0, 1].set_xlabel("r")
axs[0, 1].legend()
axs[0, 1].grid(alpha=0.25)

# 3) log-log rho and Sigma
mask = (r > 0.5) & (r < 16.0)
axs[0, 2].loglog(r[mask], rho[mask], label="rho")
axs[0, 2].loglog(Rproj[mask], Sigma[mask], label="Sigma(R)")
axs[0, 2].set_title("log-log density / projected density")
axs[0, 2].set_xlabel("r or R")
axs[0, 2].legend()
axs[0, 2].grid(alpha=0.25)

# 4) history
axs[1, 0].plot(hist_step, hist_slope_rho, label="slope_rho")
axs[1, 0].plot(hist_step, hist_slope_M, label="slope_M")
axs[1, 0].plot(hist_step, hist_slope_Sigma, label="slope_Sigma")
axs[1, 0].plot(hist_step, hist_flatness_v, label="flatness_v")
axs[1, 0].plot(hist_step, hist_shell, label="shell_ratio")
axs[1, 0].set_title("dynamics history")
axs[1, 0].set_xlabel("step")
axs[1, 0].legend()
axs[1, 0].grid(alpha=0.25)

# 5) main lensing diagnostics
ldm = lens_by_geom[SIGMA_CRIT_MAIN]
axs[1, 1].plot(Rproj, ldm["kappa"], label="kappa")
axs[1, 1].plot(Rproj, ldm["bar_kappa"], label="bar_kappa")
axs[1, 1].axhline(1.0, linestyle="--", label="critical = 1")
if not np.isnan(Rcrit_main):
    axs[1, 1].axvline(Rcrit_main, linestyle="--", label="Rcrit(main)")
if not np.isnan(R_shell):
    axs[1, 1].axvline(R_shell, linestyle=":", label="R_shell")
axs[1, 1].set_title(f"lensing mass diagnostics (Sigma_crit={SIGMA_CRIT_MAIN})")
axs[1, 1].set_xlabel("R")
axs[1, 1].legend()
axs[1, 1].grid(alpha=0.25)

# 6) tangential eigenvalue for multiple geometries
for sc in SIGMA_CRIT_LIST:
    ld = lens_by_geom[sc]
    axs[1, 2].plot(Rproj, ld["lambda_t"], label=f"lambda_t, Sc={sc}")
    if not np.isnan(ld["Rcrit"]):
        axs[1, 2].axvline(ld["Rcrit"], linestyle="--")
axs[1, 2].axhline(0.0, linestyle="--", label="critical")
if not np.isnan(R_balance):
    axs[1, 2].axvline(R_balance, linestyle=":", label="R_balance")
if not np.isnan(R_shell):
    axs[1, 2].axvline(R_shell, linestyle="-.", label="R_shell")
axs[1, 2].set_title("tangential criticality")
axs[1, 2].set_xlabel("R")
axs[1, 2].legend(fontsize=8)
axs[1, 2].grid(alpha=0.25)

plt.tight_layout()
fig.savefig(OUTDIR / "kf_v7c_summary.png", dpi=170)
plt.show()