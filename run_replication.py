"""Reproduce Tables 5-10 and 12 of the paper from the model in model.py and print them next to the published values.
Usage:  python run_replication.py [--like-for-like] [--seeds 8]
"""
import sys, numpy as np
import model
from model import *

LFL = "--like-for-like" in sys.argv
SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 3
model.set_basis(LFL)
print("Baseline:", "like-for-like (FY2026-27-only items removed)" if LFL else "CGFA totals as published")

def row(label, mine, paper):
    f = lambda v: "-" if v is None else (f"{v:8.2f}" if isinstance(v, (int, float, np.floating)) else str(v))
    print(f"  {label:26s} mine: " + " ".join(f(v) for v in mine) + "\n  " + " " * 26 + " paper: " + " ".join(f(v) for v in paper))

S0 = totals("Cbar"); P = totals("P"); L = totals("L")
pv5 = lambda C: sum(C[i] / 1.05 ** (YEARS[i] - 2025 - 0.5) for i in range(n)) / 1000
print("\n== Table 5 (FY2026, FY2035, FY2045, cumulative, PV@5%, funded %, unfunded $B) ==")
row("S0 current law", [S0[0]/1e3, S0[9]/1e3, S0[-1]/1e3, S0.sum()/1e3, pv5(S0), 90.0, (L[-1]-sum(SYSTEMS[s].Abar[-1] for s in SYS))/1e3], [11.7, 14.9, 18.6, 297.1, 182.0, 90.0, 35.3])
c1 = sum(level_contribution({s: SYSTEMS[s].Abar[-1] for s in SYS}).values()); c2 = sum(level_contribution({s: 0.8 * SYSTEMS[s].L[-1] for s in SYS}).values())
row("S1 level-dollar 90%", [c1/1e3, c1/1e3, c1/1e3, 20*c1/1e3, pv5(np.full(n, c1)), 90.0, (L[-1]-sum(SYSTEMS[s].Abar[-1] for s in SYS))/1e3], [14.0, 14.0, 14.0, 280.9, 179.4, 90.0, 35.3])
row("S2 level-dollar 80%", [c2/1e3, c2/1e3, c2/1e3, 20*c2/1e3, pv5(np.full(n, c2)), 80.0, 0.2*L[-1]/1e3], [13.2, 13.2, 13.2, 264.0, 168.6, 80.0, 70.6])
PUB5 = {"S3": [12.7, 14.4, 16.6, 291.1, 181.9, 91.1, 31.4], "S4": [11.2, 12.5, 14.2, 252.4, 158.1, 69.4, 107.9],
        "S5": [12.1, 13.6, 15.6, 274.5, 171.7, 81.8, 64.1], "S6": [13.4, 15.3, 17.8, 308.9, 192.8, 101.1, -3.9]}
for nm, kw in {"S3": dict(N=20, balloon=0.10), "S4": dict(N=30), "S5": dict(N=25), "S6": dict(N=20)}.items():
    r = sim_all("layered", **kw); C = r["C"]
    row(f"{nm} layered", [C[0]/1e3, C[9]/1e3, C[-1]/1e3, C.sum()/1e3, pv5(C), r["Ahat"][-1]/r["L"][-1]*100, r["U"][-1]/1e3], PUB5[nm])

print("\n== Table 6 step 3: July-1 payments, 80% target, cumulative $B (paper 255.5) ==")
print("  mine:", round(sum(20 * (0.8 * SYSTEMS[s].L[-1] - SYSTEMS[s].A_T_zero()) / sum((1 + SYSTEMS[s].rho) ** (T - t + 1) for t in YEARS) for s in SYS) / 1e3, 1))

print("\n== Table 7 (FY2026, FY2026-35 avg, FY2045, funded %, unfunded $B) ==")
PUB7 = {15: [15.8, 17.0, 1.3, 102.0, -7.0], 20: [13.4, 14.4, 17.8, 101.1, -3.9], 25: [12.1, 12.8, 15.6, 81.8, 64.1], 30: [11.2, 11.9, 14.2, 69.4, 107.9], 35: [10.6, 11.2, 13.3, 60.9, 137.8], 40: [10.2, 10.7, 12.6, 54.9, 159.1]}
for N, pub in PUB7.items():
    r = sim_all("layered", N=N); row(f"ADC {N}-year", [r["C"][0]/1e3, r["C"][:10].mean()/1e3, r["C"][-1]/1e3, r["Ahat"][-1]/r["L"][-1]*100, r["U"][-1]/1e3], pub)

print("\n== Table 8: one-time loss = assumed return - 27 points (peak extra $B, % payroll, largest 1-yr increase $B, funded %) ==")
PUB8 = {2030: ([6.4, 16, 2.2, 90], [4.9, 12, 1.0, 83]), 2036: ([12.0, 30, 3.1, 90], [5.5, 14, 1.3, 75]), 2040: ([35.6, 87, 15.0, 90], [5.9, 15, 1.5, 71]), 2043: ([27.2, 67, 19.9, 85], [2.6, 6, 1.8, 76])}
for y in PUB8:
    for k, (nm, rl, kw) in enumerate([("current law", "statutory", {}), ("layered 90%", "layered", dict(N=20, balloon=0.10))]):
        base = sim_all(rl, **kw); C = np.zeros(n); A = np.zeros(n)
        for s in SYS:
            rets = np.full(n, SYSTEMS[s].rho); rets[YEARS.index(y)] = SYSTEMS[s].rho - 0.27
            r = sim_system(s, rl, rets=rets, **kw); C += r["C"]; A += r["Ahat"]
        ex = C - base["C"]; inc = np.diff(np.concatenate([[base["C"][0]], C]))
        row(f"FY{y} {nm}", [ex.max()/1e3, (ex/P).max()*100, inc.max()/1e3, A[-1]/L[-1]*100], PUB8[y][k])

print("\n== Table 9: permanent return shortfall (FY2035, FY2045 $B, FY2045 % payroll, cumulative $B, funded %) ==")
PUB9 = {(0.01, "CL"): [16.1, 28.4, 69, 342.1, 89], (0.01, "LAY"): [15.4, 19.8, 48, 315.5, 83], (0.01, "A25"): [14.5, 18.6, 46, 298.1, 74],
        (0.02, "CL"): [17.2, 37.0, 91, 382.3, 88], (0.02, "LAY"): [16.3, 22.6, 55, 337.6, 75], (0.02, "A25"): [15.4, 21.2, 52, 319.4, 67]}
for (sf, key), pub in PUB9.items():
    rl, kw = {"CL": ("statutory", {}), "LAY": ("layered", dict(N=20, balloon=0.10)), "A25": ("layered", dict(N=25))}[key]
    r = sim_all(rl, shortfall=sf, **kw); row(f"-{sf*100:.0f}pt {key}", [r["C"][9]/1e3, r["C"][-1]/1e3, r["C"][-1]/r["P"][-1]*100, r["C"].sum()/1e3, r["Ahat"][-1]/r["L"][-1]*100], pub)

print("\n== Table 12: dedicated revenue (cumulative added, unfunded FY2045 $B, funded %, year 80%, year 90%) ==")
share = {s: SYSTEMS[s].Cbar[0] / sum(SYSTEMS[x].Cbar[0] for x in SYS) for s in SYS}
PUB12 = {0.0: ["-", 35.3, 90.0, 2043, 2045], 0.25: [5.4, 25.3, 92.8, 2042, 2045], 0.50: [10.7, 15.4, 95.6, 2042, 2044], 0.685: [14.7, 8.0, 97.7, 2041, 2044], 1.0: [21.4, -4.5, 101.3, 2041, 2043], 2.0: [42.8, -44.2, 112.5, 2039, 2042]}
for R, pub in PUB12.items():
    Rt = np.array([R * 1.02 ** (t - 2028) if t >= 2028 else 0.0 for t in YEARS]); Ah = np.zeros(n)
    for s in SYS: Ah += sim_system(s, "baseline", extra=Rt * share[s] * 1000)["Ahat"]
    fr = Ah / L; y80 = next((YEARS[i] for i in range(n) if fr[i] >= .8), None); y90 = next((YEARS[i] for i in range(n) if fr[i] >= .9), None)
    row(f"R2028 = {R:.3f}B", [Rt.sum(), (L[-1]-Ah[-1])/1e3, fr[-1]*100, y80, y90], pub)

print(f"\n== Table 10: Monte Carlo, {SEEDS} seeds x 10,000 paths (mean over seeds) ==")
PUB10 = {"Current law": [91.5, 83.9, 0.5, 3.1, 22.8, 37.0, 63, 54.2, 20.5, 14.6, 61.0, 286.0], "Layered 90%": [92.8, 60.6, 29.0, 1.2, 2.5, 0.0, 50, 17.5, 0.0, 15.8, 27.3, 282.5],
         "ADC 25": [83.3, 53.3, 44.1, 1.1, 2.3, 0.0, 47, 11.3, 0.0, 14.8, 25.8, 266.2], "ADC 30": [70.7, 43.7, 65.8, 1.0, 2.1, 0.0, 43, 5.0, 0.0, 13.5, 23.8, 244.7]}
names = ["funded_med", "funded_p5", "P_funded<80", "inc_med", "inc_p95", "P_inc>5", "peak_med", "P_peak>60", "P_peak>100", "c45_med", "c45_p95", "cum_med"]
print("  stats:", ", ".join(names))
for key, (rl, kw) in RULES.items():
    res = []
    for sd in range(SEEDS):
        draws = np.random.default_rng(11 * (sd + 1)).integers(0, 21, size=(10000, n)); C, Ah = mc_all(rl, draws, **kw); res.append(mc_metrics(C, Ah))
    row(key, [np.mean([r[nm] for r in res]) for nm in names], PUB10[key])
print("\nDone.")
