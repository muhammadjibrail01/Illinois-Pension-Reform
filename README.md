# Replication package: "Stabilizing Illinois' Pension System"

Rebuilds the model in Section 5 of the paper and reproduces Tables 5-10 and 12 and the Table A2 baseline.

## Run
    pip install numpy scipy
    python run_replication.py                 # CGFA totals as published (the paper's basis)
    python run_replication.py --like-for-like # FY2026-27-only items removed from the baseline (paper, Section 4.2)
    python run_replication.py --seeds 8       # more Monte Carlo seeds (default 3)
`expected_output.txt` is the output of `python run_replication.py --seeds 2`.

## Files
* `model.py` - the model: Eqs. 1-12, the statutory, layered and level-payment rules, and the Monte Carlo.
* `run_replication.py` - prints every replicated table beside the published values.
* `baseline.json` - CGFA Special Pension Briefing (Nov. 2025) FY2026-45 projections by system, and FY2025 starting balances.
* `returns_appendixP.csv` - 2003 pension-bond net returns FY2004-24 (TRS, SURS, ISBI), CGFA FY2024 Financial Condition report, Appendix P.

## Conventions needed to reproduce the published numbers
1. State and employee flows occur mid-year and earn half a year of return (Eq. 1). This reproduces the final SERS and SURS projected assets to 0.005 percent from FY2031.
2. The net benefit outflow is inferred from the CGFA actuarial-asset path (Eq. 4), so S0 reproduces the CGFA baseline exactly.
3. Layered rules: payroll growth `g` in Eq. 10 is each system's own average projected payroll growth over FY2026-45 (TRS 2.4%, SERS 2.1%, SURS 2.8%, JRS 1.6%, GARS 1.2%).
4. Employer normal cost: CGFA FY2024-vintage combined path, scaled to the published FY2027 value ($2,556.9 million) and split across systems by FY2027 shares.
5. Table 8 loss: assumed return minus 27 points (-20% for TRS). A flat -20% gives results about 0.5% lower.
6. Monte Carlo: bootstrap of the 21 averaged bond returns (geometric mean 7.40%), shifted so the geometric mean is 6.75%, applied as deviations from each system's assumed return, common to all systems; five-year smoothing (Eqs. 2-3).
7. Present values discount mid-year flows to FY2025 (Table 5 at 5%; Section 6.3 at a flat 6.8%).

## What is and is not reproduced
* Tables 5-7, 9, 12 and A2: identical to the published rounding. Table 8: identical to rounding (item 5).
* Table 10: random draws differ, so statistics agree within sampling error (probabilities within about one point; dollar medians within about 1.5 percent).
* Not reproduced: Section 6.8's General Funds shares (CGFA's General Funds chart, Appendix V, is an image) and Figures 1-3.

## Known feature of the CGFA baseline
CGFA's FY2026-27 State contributions include SERS bond debt service ($127.6M, $130.5M) and SURS retirement-savings-plan, excess-benefit and deferred-compensation amounts ($135.7M, $144.9M) that later years exclude. `--like-for-like` removes them; no cumulative, FY2045 or simulated result changes by more than 0.5 percent, but FY2026 comparisons of the layered rules with current law do (paper, Section 4.2).
