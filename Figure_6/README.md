### README (Figure 6)

Executing "Fig6.py" computes the maximum sustainable cell volume V_max against growth rate for a spherocylinder with the aspect ratio of E. coli (total length 4 diameters), without and with the intracellular diffusion penalty, and the proteome allocation against cell volume at the overflow transition. It writes Fig6A.pdf, Fig6BC.pdf and fig6_data.json to "images". Executing "compose_fig6.py" afterwards assembles the three-panel Fig6.pdf, and a version of panel A with the spherical-cell curves (Fig6A_with_sphere.pdf) that is used in the response to reviewers. Run both from this folder; Fig6.py takes about ten minutes.

Model functions are in "methodsFig6.py" with the linear flux-balance solver "fluxbalance_linear.py" (see the note in ../Figures_2_and_3/README.md). The import capacity is K_C = 52 uM.

Procedure, as in Methods 5.2 of the manuscript:
- Geometry: V = 22 pi R^3/3, S = 16 pi R^2, so S = (9216 pi/121)^(1/3) V^(2/3) and phi_E = (337.5/M)^(1/3) with M in MDa; diffusion path 8R - 2d.
- For each target growth rate, glucose is throttled by bisection so that the realized growth rate equals the target; V_max is the largest protein mass M for which all sector fractions are non-negative and the proteome budget is met.
- The diffusion penalty beta/T is normalized to unity at the empirical cell of the same growth rate (V = 0.17 exp(1.16 kappa)).

"size_vs_growth_data.csv" holds the newborn volumes of E. coli NCM3722 plotted in panel A (Taheri-Araghi et al. 2015 and Si et al. 2017 as compiled in Serbanescu et al. 2020; newborn volume is the reported cycle-averaged volume times ln 2). Columns: kappa_h, V_um3, medium, medium_type, source.

Code variables map to manuscript symbols as in the other folders (phiF -> phi_G, eF -> e_G, epsilonF -> epsilon_G, phiOMaxFrac -> gamma, beta -> beta', betaT -> beta/T).
