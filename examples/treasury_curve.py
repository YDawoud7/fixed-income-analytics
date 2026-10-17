"""Build and analyse the US Treasury curve for one day.

    python examples/treasury_curve.py                      # latest available day
    python examples/treasury_curve.py --date 2025-12-31
    python examples/treasury_curve.py --csv examples/sample_par_curve.csv   # offline

Steps: download par yields, bootstrap a zero curve, fit Nelson-Siegel-Svensson
and Nelson-Siegel to the zero rates, then price and risk a 10-year note off the curve.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from fixedincome.bond import FixedRateBond
from fixedincome.bootstrap import bootstrap_par_curve
from fixedincome.curve_risk import key_rate_durations, z_spread
from fixedincome.data.treasury import fetch_par_curve, parse_par_curve_csv
from fixedincome.nss import fit_nelson_siegel, fit_svensson
from fixedincome.schedule import add_months


def load_row(args) -> tuple[pd.Timestamp, pd.Series]:
    if args.csv:
        df = parse_par_curve_csv(Path(args.csv).read_text())
    else:
        year = int(args.date[:4]) if args.date else date.today().year
        df = fetch_par_curve(year, cache_dir="data_cache")
    if args.date:
        df = df.loc[: args.date]
    row = df.iloc[-1].dropna()
    return df.index[-1], row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="YYYY-MM-DD (uses the last available day on or before it)")
    p.add_argument("--csv", help="parse a saved Treasury CSV instead of downloading")
    p.add_argument("--plot", default="figures/curve.png", help="output path for the chart ('' to skip)")
    args = p.parse_args()

    asof, row = load_row(args)
    tenors, pars = row.index.to_numpy(float), row.to_numpy(float)
    curve = bootstrap_par_curve(tenors, pars)

    # Fit the parametric models to the bootstrapped zero rates at the pillars.
    zeros = np.asarray(curve.zero_rate(tenors))
    ns = fit_nelson_siegel(tenors, zeros)
    nss = fit_svensson(tenors, zeros)

    print(f"US Treasury curve, {asof:%Y-%m-%d}")
    print(f"{'tenor':>6} {'par %':>7} {'zero %':>7} {'NSS %':>7} {'1y fwd %':>9}   (zero/fwd: continuous)")
    for t, y in zip(tenors, pars):
        fwd = curve.forward_rate(t, t + 1)
        print(f"{t:6.2f} {100*y:7.3f} {100*curve.zero_rate(t):7.3f} {100*float(nss(t)):7.3f} {100*fwd:9.3f}")
    print(
        f"\nNelson-Siegel: b0={ns.b0:.4f} b1={ns.b1:.4f} b2={ns.b2:.4f} tau={ns.tau:.2f}  rmse={ns.rmse*1e4:.2f}bp"
        f"\nSvensson:      b0={nss.b0:.4f} b1={nss.b1:.4f} b2={nss.b2:.4f} b3={nss.b3:.4f} "
        f"tau1={nss.tau1:.2f} tau2={nss.tau2:.2f}  rmse={nss.rmse*1e4:.2f}bp"
    )

    # A hypothetical 10-year note priced at par on the curve's 10y par yield.
    settle = asof.date()
    coupon = round(float(row[10.0]) * 800) / 800  # nearest 1/8 of a percent
    note = FixedRateBond(coupon=coupon, maturity=add_months(settle, 120))
    ytm = float(row[10.0])
    clean = note.clean_price(ytm, settle)
    print(f"\n10y note {100*coupon:.3f}% at {100*ytm:.3f}%: clean {clean:.4f}")
    print(f"  modified duration {note.modified_duration(ytm, settle):.3f}, convexity {note.convexity(ytm, settle):.2f}, "
          f"DV01 {note.dv01(ytm, settle):.4f}")
    print(f"  z-spread to bootstrapped curve: {1e4*z_spread(note, curve, settle, clean):.2f}bp")
    krd = key_rate_durations(note, curve, settle)
    print("  key rate durations: " + ", ".join(f"{k:g}y {v:.3f}" for k, v in krd.items() if abs(v) > 1e-6))

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        fine = np.linspace(0.08, 30, 300)
        ax.plot(tenors, 100 * pars, "o", color="#1f3b73", label="Par yields (Treasury)")
        ax.plot(fine, 100 * np.asarray(curve.zero_rate(fine)), color="#2a9d8f", label="Bootstrapped zero")
        ax.plot(fine, 100 * ns(fine), ":", color="#6c757d", label="Nelson-Siegel fit")
        ax.plot(fine, 100 * nss(fine), "--", color="#e76f51", label="Svensson fit")
        # Forwards from the bootstrapped curve inherit kinks from interpolating par
        # yields, so plot the smooth forward implied by the Svensson fit instead.
        t1 = fine[fine <= 29]
        fwd = ((t1 + 1) * nss(t1 + 1) - t1 * nss(t1)) * 100
        ax.plot(t1, fwd, color="#b5838d", lw=1, alpha=0.8, label="1y forward (Svensson)")
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Rate (%)")
        ax.set_title(f"US Treasury curve, {asof:%d %b %Y}")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        out = Path(args.plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"\nSaved chart to {out}")


if __name__ == "__main__":
    main()
