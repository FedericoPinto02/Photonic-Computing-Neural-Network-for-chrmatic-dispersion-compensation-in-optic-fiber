# ONN-Based Chromatic Dispersion Pre-Compensation (Simulation)

A compact MATLAB implementation of the optical neural network (ONN) pre-compensation experiment reported in our reference paper. The project models a silicon-photonic perceptron (N-tap delay–weight–sum) operating on the **optical field** before fiber propagation (IM–DD, PAM4), and reproduces the paper’s workflow end-to-end: ONN shaping → dispersive fiber → optical/electrical filtering → direct detection → alignment → metrics (BER/eye, loss).

---

## Project workflow (what happens and in what order)

1. **Waveform generation.** Create a PAM4 PRBS at the target baud rate with oversampling.
2. **ONN pre-compensation.** Build \(N\) delayed replicas (tap spacing \(\Delta t\)) and apply per-tap complex (or phase-only) weights derived from the MZI model; coherently sum to get the pre-shaped field.
3. **Channel.** Propagate through a dispersive fiber (chromatic dispersion as quadratic phase), apply an optical band-pass, and inject ASE noise to meet a target OSNR (0.1 nm reference).
4. **Receiver.** Perform square-law photodetection and electrical low-pass filtering.
5. **Alignment & sampling.** Align the received trace to the known PRBS by cross-correlation and extract symbol-time samples (used for loss/BER).
6. **Training & evaluation.** Optimize ONN weights with a margin-based \(L_2\) loss (PSO, optional Adam refinement). Report BER via a Gaussian-MAP estimator, eye metrics, and sensitivity vs PRX/OSNR and distance.


---

## Quick start

1. Open MATLAB (R2021+ recommended).  
2. Set key params in `main.m` (baud, `N`, `Δt`, span `L`, RX/optical BW, OSNR).  
3. Run:
   
