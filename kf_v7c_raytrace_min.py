import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# KF v7c.2 : minimal axisymmetric ray tracing
#
# Load lens profile from kf_v7c_final.npz
# Build circular lens equation:
#   beta_vec = theta_vec - alpha(theta) * e_theta
#
# In dimensionless axisymmetric lensing:
#   alpha(R) = bar_kappa(<R) * R
# so:
#   beta = R * (1 - bar_kappa(<R)) = R * lambda_t
#
# This is enough to produce:
#   - Einstein ring (aligned source)
#   - arcs / double image tendency (offset source)
# ============================================================

# ----------------------------
# paths
# ----------------------------
NPZ_PATH = Path("kf_v7c_outputs/kf_v7c_final.npz")
OUTDIR = Path("kf_v7c_outputs/raytrace")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# choose geometry
# must match one of your saved Sigma_crit values
# e.g. 0.350 / 0.550 / 0.800
# ----------------------------
SIGMA_CRIT_TAG = "0.550"

# ----------------------------
# source settings
# ----------------------------
# set to (0,0) for near-perfect Einstein ring
# offset it a bit for arcs / broken ring
SRC_X0 = 0.18
SRC_Y0 = 0.00
SRC_SIGMA = 0.06
SRC_AMP = 1.0

# try a second centered source if you want comparison
MAKE_CENTERED_SOURCE = True

# ----------------------------
# image plane grid
# ----------------------------
IMG_N = 700
THETA_MAX = 8.0
x = np.linspace(-THETA_MAX, THETA_MAX, IMG_N)
y = np.linspace(-THETA_MAX, THETA_MAX, IMG_N)
XX, YY = np.meshgrid(x, y)
RR = np.sqrt(XX**2 + YY**2)

EPS = 1e-12

# ----------------------------
# helpers
# ----------------------------
def gaussian_source(beta_x, beta_y, x0, y0, sigma, amp=1.0):
    return amp * np.exp(-((beta_x - x0)**2 + (beta_y - y0)**2) / (2.0 * sigma**2))

def interp1_nonneg(xq, x, y):
    return np.interp(xq, x, y, left=y[0], right=y[-1])

def make_lensed_image(R_grid, X_grid, Y_grid, Rprof, bar_kappa_prof, src_x0, src_y0, src_sigma):
    # alpha(R) = bar_kappa(<R) * R
    bar_kappa = interp1_nonneg(R_grid, Rprof, bar_kappa_prof)
    alpha = bar_kappa * R_grid

    # beta_vec = theta_vec - alpha * e_r
    # with e_r = (x/r, y/r)
    fac = np.where(R_grid > EPS, 1.0 - alpha / np.maximum(R_grid, EPS), 1.0 - bar_kappa[0])

    beta_x = fac * X_grid
    beta_y = fac * Y_grid

    img = gaussian_source(beta_x, beta_y, src_x0, src_y0, src_sigma, SRC_AMP)
    return img, beta_x, beta_y, alpha, bar_kappa

# ----------------------------
# load lens data
# ----------------------------
data = np.load(NPZ_PATH)

keys = list(data.keys())
print("Loaded keys:")
for k in keys:
    print(" ", k)

Rproj = data["Rproj"]

bar_key = f"bar_kappa_{SIGMA_CRIT_TAG}"
kappa_key = f"kappa_{SIGMA_CRIT_TAG}"
lam_t_key = f"lambda_t_{SIGMA_CRIT_TAG}"

if bar_key not in data:
    raise KeyError(f"Missing {bar_key}. Available keys: {keys}")

bar_kappa_prof = data[bar_key]
kappa_prof = data[kappa_key] if kappa_key in data else None
lambda_t_prof = data[lam_t_key] if lam_t_key in data else (1.0 - bar_kappa_prof)

# critical radius from lambda_t ~ 0
i_crit = np.argmin(np.abs(lambda_t_prof))
Rcrit = Rproj[i_crit]
print(f"\nChosen Sigma_crit = {SIGMA_CRIT_TAG}")
print(f"Estimated Rcrit ~ {Rcrit:.4f}")
print(f"min |lambda_t| ~ {np.abs(lambda_t_prof[i_crit]):.6f}")

# ----------------------------
# ray trace for offset source
# ----------------------------
img_offset, beta_x_off, beta_y_off, alpha_grid, bar_grid = make_lensed_image(
    RR, XX, YY, Rproj, bar_kappa_prof, SRC_X0, SRC_Y0, SRC_SIGMA
)

# centered source for clean ring
if MAKE_CENTERED_SOURCE:
    img_center, beta_x_ctr, beta_y_ctr, _, _ = make_lensed_image(
        RR, XX, YY, Rproj, bar_kappa_prof, 0.0, 0.0, SRC_SIGMA
    )

# ----------------------------
# build source-plane visualization grid
# ----------------------------
BMAX = 1.2
bx = np.linspace(-BMAX, BMAX, 400)
by = np.linspace(-BMAX, BMAX, 400)
BXX, BYY = np.meshgrid(bx, by)

src_offset = gaussian_source(BXX, BYY, SRC_X0, SRC_Y0, SRC_SIGMA, SRC_AMP)
if MAKE_CENTERED_SOURCE:
    src_center = gaussian_source(BXX, BYY, 0.0, 0.0, SRC_SIGMA, SRC_AMP)

# ----------------------------
# radial image profile
# ----------------------------
def radial_mean(img, R_grid, nbins=240):
    rmax = np.max(R_grid)
    bins = np.linspace(0, rmax, nbins + 1)
    rc = 0.5 * (bins[:-1] + bins[1:])
    prof = np.zeros(nbins)
    for i in range(nbins):
        m = (R_grid >= bins[i]) & (R_grid < bins[i + 1])
        prof[i] = np.mean(img[m]) if np.any(m) else 0.0
    return rc, prof

rprof_off, iprof_off = radial_mean(img_offset, RR, nbins=260)
if MAKE_CENTERED_SOURCE:
    rprof_ctr, iprof_ctr = radial_mean(img_center, RR, nbins=260)

# ----------------------------
# plots
# ----------------------------
ncols = 3 if MAKE_CENTERED_SOURCE else 2
fig, axs = plt.subplots(2, ncols, figsize=(5 * ncols, 9))

# row 1: source plane
axs[0, 0].imshow(
    src_offset,
    extent=[-BMAX, BMAX, -BMAX, BMAX],
    origin="lower",
    aspect="equal",
)
axs[0, 0].set_title("source plane (offset source)")
axs[0, 0].set_xlabel("beta_x")
axs[0, 0].set_ylabel("beta_y")

if MAKE_CENTERED_SOURCE:
    axs[0, 1].imshow(
        src_center,
        extent=[-BMAX, BMAX, -BMAX, BMAX],
        origin="lower",
        aspect="equal",
    )
    axs[0, 1].set_title("source plane (centered source)")
    axs[0, 1].set_xlabel("beta_x")
    axs[0, 1].set_ylabel("beta_y")

# row 1 last: lens radial diagnostics
ax_diag = axs[0, -1]
if kappa_prof is not None:
    ax_diag.plot(Rproj, kappa_prof, label="kappa")
ax_diag.plot(Rproj, bar_kappa_prof, label="bar_kappa")
ax_diag.plot(Rproj, lambda_t_prof, label="lambda_t")
ax_diag.axhline(1.0, linestyle="--", linewidth=1, label="critical=1 (for bar_kappa)")
ax_diag.axhline(0.0, linestyle=":", linewidth=1, label="lambda_t=0")
ax_diag.axvline(Rcrit, linestyle="--", linewidth=1.2, label=f"Rcrit~{Rcrit:.2f}")
ax_diag.set_xlim(0, min(THETA_MAX, np.max(Rproj)))
ax_diag.set_title(f"lens diagnostics (Sigma_crit={SIGMA_CRIT_TAG})")
ax_diag.set_xlabel("R")
ax_diag.legend(fontsize=8)
ax_diag.grid(alpha=0.25)

# row 2: image plane
axs[1, 0].imshow(
    img_offset,
    extent=[-THETA_MAX, THETA_MAX, -THETA_MAX, THETA_MAX],
    origin="lower",
    aspect="equal",
)
axs[1, 0].add_patch(plt.Circle((0, 0), Rcrit, fill=False, linestyle="--", linewidth=1.2))
axs[1, 0].set_title("lensed image (offset source)")
axs[1, 0].set_xlabel("theta_x")
axs[1, 0].set_ylabel("theta_y")

if MAKE_CENTERED_SOURCE:
    axs[1, 1].imshow(
        img_center,
        extent=[-THETA_MAX, THETA_MAX, -THETA_MAX, THETA_MAX],
        origin="lower",
        aspect="equal",
    )
    axs[1, 1].add_patch(plt.Circle((0, 0), Rcrit, fill=False, linestyle="--", linewidth=1.2))
    axs[1, 1].set_title("lensed image (centered source)")
    axs[1, 1].set_xlabel("theta_x")
    axs[1, 1].set_ylabel("theta_y")

# row 2 last: radial brightness
ax_prof = axs[1, -1]
ax_prof.plot(rprof_off, iprof_off, label="offset source")
if MAKE_CENTERED_SOURCE:
    ax_prof.plot(rprof_ctr, iprof_ctr, label="centered source")
ax_prof.axvline(Rcrit, linestyle="--", linewidth=1.2, label=f"Rcrit~{Rcrit:.2f}")
ax_prof.set_xlim(0, THETA_MAX)
ax_prof.set_title("radial image brightness")
ax_prof.set_xlabel("image-plane radius")
ax_prof.legend()
ax_prof.grid(alpha=0.25)

plt.tight_layout()
fig.savefig(OUTDIR / f"raytrace_sigmaCrit_{SIGMA_CRIT_TAG}.png", dpi=180)
plt.show()

# ----------------------------
# print quick interpretation
# ----------------------------
print("\n==== quick interpretation ====")
print("1) centered source should produce a near-ring at radius ~ Rcrit")
print("2) offset source should break ring symmetry into brighter arc-like sides")
print("3) if the bright ring/arc sits near Rcrit, your KF-grown lens really crossed the tangential critical condition")