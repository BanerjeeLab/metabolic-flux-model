# metabolic-flux-model

Computational metabolic flux model for a growing bacterium, coupling cell shape, proteome allocation and intracellular diffusion, as detailed in: "Bacterial biosynthetic efficiency is constrained by cell geometry and intracellular diffusion", A. Cylke, I. Badvaram and S. Banerjee (PLOS Computational Biology).

The model partitions the proteome into ribosomal, envelope, glycolytic and housekeeping sectors and solves steady-state balances for carbon, acetyl-CoA and ATP. Respiration is limited by the envelope area, and internal fluxes can be attenuated by an intracellular diffusion penalty that depends on cell size and cytoplasmic density.

This repository is organized by figure. Each folder contains the scripts for one figure, its own README with run instructions, output file names and the code-to-manuscript notation map, and an "images" subfolder with the panels as generated and the composed manuscript figure.

- `Figures_2_and_3/` growth laws without diffusion: proteome allocation, growth rate, acetate excretion and efficiency against glucose (Fig. 2) and against growth rate and surface-to-volume ratio (Fig. 3).
- `Figure_4/` volume and surface perturbations of the growth-law cell.
- `Figure_5/` efficiency against cell shape with and without the diffusion penalty.
- `Figure_6/` maximum sustainable cell volume against growth rate for a spherocylinder, with the measured newborn volumes used in panel A.
- `SupplementaryFig1/` to `SupplementaryFig5/` the supplementary figures: K_C sensitivity of Figs. 2 and 3 (S1), K_C and biosynthetic-cost sensitivity of Fig. 6 (S2), smooth respiratory saturation in place of the hard cap (S3), emergent cell size and shape (S4), nutrient shifts (S5).
- `compose_panels.py` assembles the multi-panel PDFs of Figs. 2 to 5 and S1 from the SVG panels (needs rsvg-convert and pypdf). Fig. 6 is assembled by `Figure_6/compose_fig6.py`.

Every script is run from its own folder (`cd Figure_6 && python3 Fig6.py`). Dependencies: Python 3 with numpy, scipy, matplotlib and sympy. The scripts for Figs. 2 to 5 take 10 to 50 minutes each; Fig. 6 and S2 about ten minutes; S3 to S5 under a minute.

In every folder, `fluxbalance_linear.py` solves the three balances directly as a linear system and is installed into the figure's methods module at import; the biomass carbon flux is converted to carbon atoms (CF = 1e6/12.011 atoms per MDa). Parameters are listed in Table 1 of the paper; the import capacity is K_C = 52 uM throughout.
