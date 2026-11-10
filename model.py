"""
Illinois pension funding model -- replication of Section 5 of the paper.

Inputs (both included):
  baseline.json         CGFA (Nov. 2025) FY2026-45 projections for TRS, SERS, SURS, JRS, GARS and FY2025 starting balances
  returns_appendixP.csv  2003 pension-bond returns FY2004-24 (TRS, SURS, ISBI), CGFA FY2024 Financial Condition report, Appendix P

Conventions that reproduce the published tables (see README):
  * Equation (1): State and employee flows occur mid-year and earn half a year of return.
  * Equation (4): net benefit outflow B_t is inferred from the CGFA actuarial-asset path.
  * Layered rules: payroll growth g = each system's own average projected payroll growth, FY2026-45.
  * Deterministic loss (Table 8): return = assumed return - 27 points (-20% for TRS).
  * Monte Carlo: bootstrap of the 21 averaged bond returns, recentred to a 6.75% geometric mean, applied as
    deviations from each system's assumed return, common to all systems.
"""
import json, os, csv
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "baseline.json")))
B, START = D["B"], D["START"]
SYS = ["TRS", "SERS", "SURS", "JRS", "GARS"]
YEARS = list(range(2026, 2046)); T = 2045; n = len(YEARS)
SM = [0.8, 0.6, 0.4, 0.2]                      # unrecognised share of a gain/loss 0,1,2,3 years old

# employer normal cost, FY2024-vintage combined path (CGFA FY2024 report, Table 5), scaled to the published FY2027 value
NCPATH = np.array([2481.0, 2453.9, 2416.8, 2373.4, 2326.9, 2275.5, 2217.1, 2153.1, 2084.6, 2008.7,
                   1926.0, 1838.4, 1743.7, 1643.3, 1540.3, 1440.3, 1348.6, 1268.2, 1205.3, 1165.8])
NC27 = {"TRS": 1390.9, "SERS": 636.1, "SURS": 501.2, "JRS": 26.8, "GARS": 1.8}
PUBLISHED_NC27 = 2556.9
# items the CGFA includes in FY2026-27 State contributions only ($M): SERS bond debt service; SURS RSP + EBA + DCP expense
PASS = {"SERS": (127.6, 130.5), "SURS": (114.183 + 21.000 + 0.468, 121.382 + 23.000 + 0.478)}

def arr(s, f): return np.array([B[s][str(y)][f] for y in YEARS], dtype=float)

class System:
    def __init__(self, s, like_for_like=False):
        self.s = s; self.rho = START[s]["rho"]
        self.P = arr(s, "payroll"); self.Cbar = arr(s, "state"); self.E = arr(s, "emp")
        self.L = arr(s, "aal"); self.Abar = arr(s, "ava"); self.L0 = START[s]["aal"]; self.A0 = START[s]["ava"]
        if like_for_like and s in PASS:
            self.Cbar = self.Cbar.copy(); self.Cbar[0] -= PASS[s][0]; self.Cbar[1] -= PASS[s][1]
        r = self.rho; prev = np.concatenate([[self.A0], self.Abar[:-1]])
        self.Bt = self.E + self.Cbar - (self.Abar - prev * (1 + r)) * (1 + r) ** -0.5          # Equation (4)
        self.g = (self.P[-1] / self.P[0]) ** (1 / 19) - 1                                       # payroll growth used in layers
        tot = NCPATH * (PUBLISHED_NC27 / NCPATH[1]); self.NC = tot * NC27[s] / sum(NC27.values())
    def roll(self, C):
        r = self.rho; a = self.A0; A = np.zeros(n)
        for i in range(n):
            a = a * (1 + r) + (self.E[i] - self.Bt[i] + C[i]) * (1 + r) ** 0.5; A[i] = a
        return A
    def A_T_zero(self): return self.roll(np.zeros(n))[-1]

SYSTEMS = {}
def set_basis(like_for_like=False):
    SYSTEMS.clear(); SYSTEMS.update({s: System(s, like_for_like) for s in SYS})
set_basis(False)

def phi(nyrs, g, rho):                                                                          # Equation (10)
    if nyrs <= 0: return 0.0
    q = (1 + g) / (1 + rho); return 1.0 / (sum(q ** j for j in range(nyrs)) / (1 + rho))

def totals(f): return sum(getattr(SYSTEMS[s], f) for s in SYS)

def sim_system(s, rule, rets=None, N=25, balloon=0.0, newyrs=20, shortfall=0.0, extra=None):
    """rule: 'baseline' | 'statutory' (Eqs 5-6) | 'layered' (Eqs 8-11)."""
    S = SYSTEMS[s]; rho = S.rho
    r_arr = np.full(n, rho - shortfall) if rets is None else np.asarray(rets, float)
    A = S.A0; Ahat_prev = S.A0; G = []; C = np.zeros(n); Ah = np.zeros(n); U = np.zeros(n); layers = []
    if rule == "layered":
        layers = [[(S.L0 - S.A0) - balloon * S.L[-1] * (1 + rho) ** -(T - 2025), N]]
    for i, t in enumerate(YEARS):
        if rule == "baseline": c = S.Cbar[i]
        elif rule == "statutory":
            Dm = Ahat_prev - (S.A0 if i == 0 else S.Abar[i - 1])
            denom = sum(S.P[j] * (1 + rho) ** (T - YEARS[j] + 0.5) for j in range(i, n))
            c = max(S.NC[i], S.Cbar[i] - Dm * (1 + rho) ** (T - t + 1) / denom * S.P[i])
        else:
            X = [W * phi(nr, S.g, rho) for W, nr in layers]
            c = max(S.NC[i], S.NC[i] + (1 + rho) ** -0.5 * sum(X))
        if extra is not None: c += extra[i]
        r = r_arr[i]
        A_new = A * (1 + r) + (S.E[i] - S.Bt[i] + c) * (1 + r) ** 0.5                            # Equation (1)
        G.insert(0, A_new - (A * (1 + rho) + (S.E[i] - S.Bt[i] + c) * (1 + rho) ** 0.5))         # Equation (3)
        Ahat = A_new - sum(SM[k] * G[k] for k in range(min(4, len(G))))                          # Equation (2)
        if rule == "layered":
            u = S.L[i] - Ahat; bal_t = balloon * S.L[-1] * (1 + rho) ** -(T - t)
            layers = [[W * (1 + rho) - x, nr - 1] for (W, nr), x in zip(layers, X)]
            layers = [l for l in layers if l[1] > 0]; layers.append([u - bal_t - sum(W for W, _ in layers), newyrs])
        A = A_new; Ahat_prev = Ahat; C[i] = c; Ah[i] = Ahat; U[i] = S.L[i] - Ahat
    return dict(C=C, Ahat=Ah, U=U)

def sim_all(rule, **kw):
    tot = dict(C=np.zeros(n), Ahat=np.zeros(n), U=np.zeros(n))
    for s in SYS:
        r = sim_system(s, rule, **kw)
        for k in tot: tot[k] += r[k]
    tot["L"] = totals("L"); tot["P"] = totals("P"); return tot

def level_contribution(target):            # Equation (7); target: dict of FY2045 assets per system -> per-system constant c
    return {s: (target[s] - SYSTEMS[s].A_T_zero()) / sum((1 + SYSTEMS[s].rho) ** (T - t + 0.5) for t in YEARS) for s in SYS}

# --------------------------- Monte Carlo ---------------------------
def bond_returns():
    rows = list(csv.DictReader(open(os.path.join(HERE, "returns_appendixP.csv"))))
    R = np.array([[float(r["TRS"]), float(r["SURS"]), float(r["ISBI"])] for r in rows]) / 100
    avg = R.mean(axis=1)
    delta = brentq(lambda d: np.prod(1 + avg - d) ** (1 / len(avg)) - 1.0675, -0.2, 0.2)
    return avg, (avg - delta) - 0.0675                     # raw averaged returns, shocks (deviations from 6.75%)
AVG, EPS = bond_returns()

def mc_system(s, rule, draws, N=25, balloon=0.0, newyrs=20):
    S = SYSTEMS[s]; rho = S.rho; M = draws.shape[0]; ret = rho + EPS[draws]
    A = np.full(M, S.A0); Ahat_prev = np.full(M, S.A0); Gh = np.zeros((M, 4)); Cs = np.zeros((M, n)); Ah = np.zeros((M, n))
    if rule == "layered":
        K = 1 + n; W = np.zeros((M, K)); W[:, 0] = (S.L0 - S.A0) - balloon * S.L[-1] * (1 + rho) ** -(T - 2025)
        nrem = np.zeros((K, n))
        for i in range(n):
            nrem[0, i] = max(N - i, 0)
            for j in range(n - 1):
                if i >= j + 1: nrem[j + 1, i] = max(newyrs - (i - (j + 1)), 0)
        PH = np.vectorize(lambda m: phi(int(m), S.g, rho) if m > 0 else 0.0)(nrem)
    for i, t in enumerate(YEARS):
        if rule == "statutory":
            Dm = Ahat_prev - (S.A0 if i == 0 else S.Abar[i - 1])
            denom = sum(S.P[j] * (1 + rho) ** (T - YEARS[j] + 0.5) for j in range(i, n))
            c = np.maximum(S.NC[i], S.Cbar[i] - Dm * (1 + rho) ** (T - t + 1) / denom * S.P[i])
        else:
            X = W * PH[:, i][None, :]; c = np.maximum(S.NC[i], S.NC[i] + (1 + rho) ** -0.5 * X.sum(axis=1))
        r = ret[:, i]; A_new = A * (1 + r) + (S.E[i] - S.Bt[i] + c) * (1 + r) ** 0.5
        Gh = np.concatenate([(A_new - (A * (1 + rho) + (S.E[i] - S.Bt[i] + c) * (1 + rho) ** 0.5))[:, None], Gh[:, :3]], axis=1)
        Ahat = A_new - (Gh * np.array(SM)[None, :]).sum(axis=1)
        if rule == "layered":
            u = S.L[i] - Ahat; W = W * (1 + rho) - X
            if i < n - 1: W[:, i + 1] = u - balloon * S.L[-1] * (1 + rho) ** -(T - t) - W.sum(axis=1)
        A = A_new; Ahat_prev = Ahat; Cs[:, i] = c; Ah[:, i] = Ahat
    return Cs, Ah

def mc_all(rule, draws, **kw):
    C = np.zeros((draws.shape[0], n)); Ah = np.zeros_like(C)
    for s in SYS:
        c, a = mc_system(s, rule, draws, **kw); C += c; Ah += a
    return C, Ah

def mc_metrics(C, Ah):
    L = totals("L"); P = totals("P"); fr = Ah[:, -1] / L[-1]
    inc = np.diff(C, axis=1).max(axis=1) / 1000; peak = (C / P[None, :]).max(axis=1) * 100
    q = lambda x, p: float(np.percentile(x, p))
    return {"funded_med": q(fr, 50) * 100, "funded_p5": q(fr, 5) * 100, "P_funded<80": (fr < 0.8).mean() * 100,
            "inc_med": q(inc, 50), "inc_p95": q(inc, 95), "P_inc>5": (inc > 5).mean() * 100,
            "peak_med": q(peak, 50), "P_peak>60": (peak > 60).mean() * 100, "P_peak>100": (peak > 100).mean() * 100,
            "c45_med": q(C[:, -1], 50) / 1000, "c45_p95": q(C[:, -1], 95) / 1000, "cum_med": q(C.sum(axis=1), 50) / 1000}

RULES = {"Current law": ("statutory", {}), "Layered 90%": ("layered", dict(N=20, balloon=0.10)),
         "ADC 25": ("layered", dict(N=25)), "ADC 30": ("layered", dict(N=30))}
