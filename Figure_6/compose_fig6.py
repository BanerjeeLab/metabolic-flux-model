"""Assemble the three panels of Fig. 6 from the data written by Fig6.py.  Run from this folder after Fig6.py."""
import os, sys, json, pathlib, numpy as np
HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE/"images"; sys.path.insert(0, str(HERE))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import Fig6 as F
main=json.load(open(OUT/'fig6_data.json'))
kap=np.array(main['kappa']); kmax=main['kappa_max']; Vemp=F.Vemp
A=lambda key: np.array([np.nan if v is None else v for v in main[key]])
plt.rcParams.update({"font.size":9.5,"axes.labelsize":11,"legend.fontsize":7.6,"axes.spines.top":False,"axes.spines.right":False,"lines.linewidth":1.8})
fig=plt.figure(figsize=(13.5,4.3)); gs=fig.add_gridspec(1,3,width_ratios=[1.25,1,1],wspace=0.32)
ax=fig.add_subplot(gs[0])
import os, csv
def panelA(ax, with_sphere):
    ax.axvspan(0.2,kmax,color="#f2f2f2",zorder=0)
    if with_sphere:
        ax.semilogy(kap,A('sphere_import'),color="#999999",lw=1.3,label="sphere, import-limited")
        ax.semilogy(kap,A('sphere_diff_emp'),color="#999999",lw=1.3,ls="--",label="sphere, diffusion-limited (same normalization)")
        ax.semilogy(kap,A('sphere_diff_ref1'),color="#bbbbbb",lw=1.0,ls=":",label="sphere, diffusion-limited, fixed-reference normalization")
    ax.semilogy(kap,A('rod_import'),color="#08519c",label="import-limited ceiling")
    ax.semilogy(kap,A('rod_diff'),color="#a63603",ls="--",label="diffusion-limited ceiling")
    pts=HERE/"size_vs_growth_data.csv"   # columns: kappa_h, V_um3, label, source
    if os.path.exists(pts):
        rows=list(csv.DictReader(open(pts)))
        mn=[r for r in rows if r['medium_type']=='carbon-limited']; aa=[r for r in rows if r['medium_type']!='carbon-limited']
        ax.semilogy([float(r['kappa_h']) for r in mn],[float(r['V_um3']) for r in mn],"o",mfc="none",mec="k",mew=1.5,ms=6.5,ls="none",label="single carbon source\n(mannose, glycerol, sorbitol, glucose)")
        ax.semilogy([float(r['kappa_h']) for r in aa],[float(r['V_um3']) for r in aa],"^",mfc="none",mec="#777777",mew=1.3,ms=6.5,ls="none",label="amino acids supplied\n($\\kappa>1.3$ h$^{-1}$: Fig. S2)")
    else:
        ax.semilogy(kap,Vemp(kap),color="k",lw=2.2,label=r"observed, $V=0.17\,e^{1.16\kappa}$")
    for x,t in [(0.757,r"$\kappa_{\rm crit}$"),(kmax,r"$\kappa_{\max}$")]:
        ax.axvline(x,color="#999999",lw=0.8,ls=":"); ax.text(x+0.006,0.075,t,fontsize=8.5,color="#666666")
    ax.text((0.2+kmax)/2,0.13,"growth on a single\ncarbon source",ha="center",fontsize=8,color="#555555")
    ax.set_xlabel(r"Growth rate $\kappa$ (h$^{-1}$)"); ax.set_ylabel(r"Max. sustainable volume $V_{\max}$ ($\mu$m$^3$)")
    ax.set_xlim(0.1,1.3); ax.set_ylim(0.05,5e2); ax.legend(loc="upper right",frameon=False,title="E. coli NCM3722 newborn volumes,\nTaheri-Araghi et al. 2015 and Si et al. 2017",title_fontsize=8.5)
panelA(ax, with_sphere=False); ax.set_title("A",loc="left",fontweight="bold")
# version with the sphere comparison curves
figR,axR=plt.subplots(figsize=(6.2,5.0)); panelA(axR, with_sphere=True); figR.tight_layout(); figR.savefig(OUT/"Fig6A_with_sphere.png",dpi=250); figR.savefig(OUT/"Fig6A_with_sphere.pdf"); plt.close(figR)
# B, C
def prof(switch,k=0.757,n=60):
    beta=F.beta_at(F.phiE_rod,Vemp(k)) if switch else 1.0; Vs=np.logspace(np.log10(0.1),np.log10(5.0),n); rows=[]
    for Vv in Vs:
        M=F.rhoM*Vv; ok,r=F.feasible(M,F.phiE_rod(M),beta,switch,k,F.eB_default)
        rows.append((Vv,r['phiR'],r['phiG'],r['phiE'],r['phiO'],r['budget'],ok) if r else (Vv,)+(np.nan,)*5+(False,))
    return np.array(rows,dtype=float)
for i,(sw,ttl,lab) in enumerate([(False,"Not diffusion-limited","B"),(True,"Diffusion-limited","C")]):
    ax=fig.add_subplot(gs[1+i]); r=prof(sw); Vv=r[:,0]; ok=r[:,6].astype(bool)
    for j,(l,c,ls) in enumerate([(r"Ribosomes $\phi_R$","#d62728","-"),(r"Glycolytic $\phi_G$","#1f77b4","-"),(r"Envelope $\phi_E$","#2ca02c","-"),(r"Respiration $\phi_O$","#2ca02c","--")]):
        ax.semilogx(Vv,r[:,1+j],color=c,ls=ls,label=l)
    ax.semilogx(Vv,r[:,5]-F.phi0,color="k",lw=1.1,label=r"$\phi_R+\phi_G+\phi_E$")
    ax.axhline(1-F.phi0,color="#888888",lw=0.8,ls=":"); ax.text(0.105,1-F.phi0+0.012,r"budget $1-\phi_0$",fontsize=7.5,color="#666666")
    if ok.any(): Vm=Vv[ok][-1]; ax.axvline(Vm,color="#a63603",lw=0.9,ls="--"); ax.text(Vm*1.07,0.02,r"$V_{\max}$",color="#a63603",fontsize=8.5)
    ax.axvline(Vemp(0.757),color="k",lw=0.8,ls=":"); ax.text(Vemp(0.757)*0.93,0.02,"observed",fontsize=7.5,ha="right")
    ax.set_title(lab,loc="left",fontweight="bold"); ax.text(0.5,1.02,ttl+r", $\kappa=\kappa_{\rm crit}$",transform=ax.transAxes,ha="center",fontsize=9.5)
    ax.set_xlabel(r"Cell volume $V$ ($\mu$m$^3$)"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_ylim(0,0.62); ax.set_xlim(0.1,5)
    if i==0: ax.set_ylabel("Proteome mass fraction"); ax.legend(loc="center left",bbox_to_anchor=(0.80,0.5),frameon=False,borderaxespad=0.0)
fig.savefig(OUT/"Fig6.pdf",bbox_inches="tight"); fig.savefig(OUT/"Fig6.png",dpi=250,bbox_inches="tight")

print("composed")
