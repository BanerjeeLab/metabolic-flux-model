"""Fig. S2: sensitivity of the maximum sustainable cell volume to the import capacity K_C (A) and to the
biosynthetic cost e_B (B), for the rod geometry of Fig. 6.  Uses the model and V_max procedure of
../Figure_6/Fig6.py.  Run from this folder: python3 FigS2.py.  Output goes to images/.
"""
import sys, json, pathlib, csv, numpy as np
HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE/"images"; OUT.mkdir(exist_ok=True)
F6 = HERE.parent/"Figure_6"; sys.path.insert(0, str(F6))
import Fig6 as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
m = F.m; fC = F.fC; ma = F.ma; rhoM = F.rhoM

if __name__ == "__main__":
    Mgrid = np.logspace(np.log10(0.003*rhoM), np.log10(3e4*rhoM), 900)
    kapS = np.round(np.arange(0.2, 2.01, 0.1), 3); si = {}
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.5))
    # (A) import capacity: diffusion-limited ceiling at four values of K_C
    for KC, c in [(30, "#9ecae1"), (52, "#08519c"), (75, "#fd8d3c"), (100, "#a63603")]:
        m.set_K_C(KC); r = F.vmax(F.phiE_rod, True, kapS, Mgrid, 'emp'); ks = F.kappa_sat()
        si[f'KC{KC}'] = ([r[k][0] for k in kapS], ks)
        axs[0].semilogy(kapS, [r[k][0] for k in kapS], color=c, ls="--", label=rf"$K_C={KC}\,\mu$M ($\kappa_{{\max}}$={ks:.2f})")
        print(f"K_C = {KC} uM: kappa_max = {ks:.3f}")
    m.set_K_C(52)
    rows = list(csv.DictReader(open(F6/"size_vs_growth_data.csv")))
    mn = [r for r in rows if r["medium_type"] == "carbon-limited"]; aa = [r for r in rows if r["medium_type"] != "carbon-limited"]
    for ax in axs:
        ax.semilogy([float(r["kappa_h"]) for r in mn], [float(r["V_um3"]) for r in mn], "o", mfc="none", mec="k", mew=1.4, ms=6, ls="none", label="single carbon source")
        ax.semilogy([float(r["kappa_h"]) for r in aa], [float(r["V_um3"]) for r in aa], "^", mfc="none", mec="#777777", mew=1.2, ms=6, ls="none", label="amino acids supplied")
    axs[0].set_xlabel(r"Growth rate $\kappa$ (h$^{-1}$)"); axs[0].set_ylabel(r"$V_{\max}$ ($\mu$m$^3$), diffusion-limited")
    axs[0].legend(frameon=False, fontsize=8.5, title="symbols: E. coli NCM3722 newborn volumes,\nTaheri-Araghi et al. 2015 and Si et al. 2017", title_fontsize=8)
    axs[0].set_title("Import capacity", fontsize=11)
    # (B) biosynthetic cost: glucose minimal medium against a translation-only cost as a stand-in for amino-acid media
    for eBv, lab, c in [(23.5, "glucose minimal: $e_B=23.5$ ATP/aa", "#a63603"), (4.0, "amino acids supplied: $e_B=4$ ATP/aa (translation only)", "#08519c")]:
        m.eBtotal = (eBv + 8.0)/(fC*ma)   # the solver reads the module-level eBtotal = (e_B + e_GAM)/(f_C m_a)
        eBc = eBv/(fC*ma); ks = F.kappa_sat(eBc)
        r = F.vmax(F.phiE_rod, True, kapS, Mgrid, 'emp', eBc); ri = F.vmax(F.phiE_rod, False, kapS, Mgrid, 'emp', eBc)
        si[f'eB{eBv}'] = ([r[k][0] for k in kapS], [ri[k][0] for k in kapS], ks)
        axs[1].semilogy(kapS, [ri[k][0] for k in kapS], color=c, label=lab + rf", $\kappa_{{\max}}$={ks:.2f}")
        axs[1].semilogy(kapS, [r[k][0] for k in kapS], color=c, ls="--")
        print(f"e_B = {eBv} ATP/aa: kappa_max = {ks:.3f}")
    m.eBtotal = (23.5 + 8.0)/(fC*ma)
    axs[1].set_xlabel(r"Growth rate $\kappa$ (h$^{-1}$)"); axs[1].set_ylabel(r"$V_{\max}$ ($\mu$m$^3$)")
    axs[1].legend(frameon=False, fontsize=8.5); axs[1].set_title("Biosynthetic cost (solid: import; dashed: diffusion)", fontsize=11)
    for ax in axs: ax.set_ylim(0.05, 2e2); ax.set_xlim(0.2, 2.0)
    fig.tight_layout(); fig.savefig(OUT/"FigS2.png", dpi=300); fig.savefig(OUT/"FigS2.pdf")
    json.dump({k: list(v) for k, v in si.items()}, open(OUT/"FigS2_data.json", "w"), indent=1, default=float)
    print("saved")
