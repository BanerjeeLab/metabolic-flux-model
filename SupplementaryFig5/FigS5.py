"""Fig. S5: morphological plasticity under nutrient perturbation: single-cell trajectories of the emergent-geometry
model (../SupplementaryFig4/FigS4.py) through a glucose upshift and downshift.  Diffusion off, eta = 4."""
import sys, pathlib, numpy as np, json
HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE/"images"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent/"SupplementaryFig4"))
import FigS4 as pg
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
uM=pg.uM; eta=4; X0kp=pg.Vemp(0.757)/0.757

def simulate_shift(G_lo, G_hi, t_shift=6.0, T=16.0, dt=2e-3, down=False):
    n=pg.nu(eta); kp=1.0; X0=X0kp*kp
    st=pg.steady(G_lo if not down else G_hi, eta, X0kp)
    V=st['Vb']; S=st['SV']*V; X=0.0; t=0.0; out=[]; Vb=V; nb=[]
    while t<T:
        G = (G_lo if t<t_shift else G_hi) if not down else (G_hi if t<t_shift else G_lo)
        M=pg.rhoM*V; phiE=pg.rhoE*S/M
        r=pg.fba(M,phiE,G); k=r['kappa']; beta=n*k*Vb**(-1/3)
        V+=k*V*dt; S+=beta*V*dt; X+=kp*V*dt; t+=dt
        out.append((t,V,S/V,k,r['JOut'],Vb))
        if X>=X0: V/=2; S/=2; X=0.0; Vb=V; nb.append((t,V))
    return np.array(out), np.array(nb), st

G_lo, G_hi = 5.0/uM, 40.0/uM
up, nb_up, st_lo = simulate_shift(G_lo,G_hi)
dn, nb_dn, st_hi = simulate_shift(G_lo,G_hi,down=True)
sh=pg.steady(G_hi,eta,X0kp)
print(f"steady low  [G0]=5 uM : kappa={st_lo['kappa']:.3f} Vb={st_lo['Vb']:.3f} S/V={st_lo['SV']:.2f} Jout={st_lo['JOut']:.2e}")
print(f"steady high [G0]=40 uM: kappa={sh['kappa']:.3f} Vb={sh['Vb']:.3f} S/V={sh['SV']:.2f} Jout={sh['JOut']:.2e}")
# relaxation time of S/V after the upshift: time to reach 90% of the change (cycle-averaged)
for lab,tr,s0,s1 in [("upshift",up,st_lo,sh),("downshift",dn,sh,st_lo)]:
    t=tr[:,0]-6.0; sv=tr[:,2]; m=t>0
    target=s1['SV']+0.1*(s0['SV']-s1['SV'])
    idx=np.where(m & ((sv-target)*(s0['SV']-s1['SV'])<0))[0]
    print(f"{lab}: S/V 90% relaxed after {t[idx[0]]:.2f} h ({t[idx[0]]*s1['kappa']/np.log(2):.1f} generations at the new rate)" if len(idx) else f"{lab}: not relaxed")

plt.rcParams.update({"font.size":10,"axes.labelsize":11,"legend.fontsize":8.5,"axes.spines.top":False,"axes.spines.right":False,"lines.linewidth":1.5})
fig,axs=plt.subplots(2,2,figsize=(9,6.2),sharex=True)
for tr,c,lab in [(up,"#08519c","upshift 5 to 40 µM"),(dn,"#a63603","downshift 40 to 5 µM")]:
    t=tr[:,0]-6.0
    axs[0,0].plot(t,tr[:,1],color=c,label=lab); axs[0,1].plot(t,tr[:,2],color=c); axs[1,0].plot(t,tr[:,3],color=c); axs[1,1].plot(t,tr[:,4]/1e8,color=c)
for ax in axs.flat: ax.axvline(0,color="k",lw=0.8,ls="--")
axs[0,0].set_ylabel(r"Cell volume $V$ ($\mu$m$^3$)"); axs[0,1].set_ylabel(r"$S/V$ ($\mu$m$^{-1}$)")
axs[1,0].set_ylabel(r"Growth rate $\kappa$ (h$^{-1}$)"); axs[1,1].set_ylabel(r"Acetate excretion $J_\mathrm{out}$ ($10^8$ h$^{-1}$)")
axs[1,0].axhline(0.757,color="grey",lw=0.8,ls=":"); axs[1,0].text(-5.8,0.77,r"$\kappa_\mathrm{crit}$",fontsize=8,color="grey")
for ax in axs[1]: ax.set_xlabel("Time after shift (h)")
axs[0,0].set_ylim(0.2,1.45); axs[0,0].legend(frameon=False,loc="upper left",ncol=2)
for ax,l in zip(axs.flat,"ABCD"): ax.set_title(l,loc="left",fontweight="bold")
fig.tight_layout(); fig.savefig(OUT/"FigS5.png",dpi=250); fig.savefig(OUT/"FigS5.pdf"); print("saved")
