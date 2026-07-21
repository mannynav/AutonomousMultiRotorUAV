A 6-DOF quadrotor flight simulator in modern C++20 with cascaded control and online parameter estimation. Built around compile-time polymorphism — C++20 concepts define the contracts, `std::variant` handles runtime dispatch, and the plant and estimator are template policies with zero runtime cost.

---

## What it does

The simulator flies a quadrotor along a reference trajectory using a two-rate cascaded controller, while a recursive least squares estimator identifies the vehicle's rotational inertias online from flight data.

```
                    ┌──────────────────────────────────────────┐
 trajectory ──────► │  OUTER LOOP  (2.5 Hz)                     │
 x, y, z, ψ refs    │  PositionController                       │
                    │  state feedback linearisation             │
                    │  → φ_ref, θ_ref, U1 (thrust)              │
                    └──────────────────┬───────────────────────┘
                                       │
                    ┌──────────────────▼───────────────────────┐
                    │  INNER LOOP  (10 Hz)                      │
                    │  RlsEstimator    → believed inertia       │
                    │  lpvDiscrete     → re-linearised model    │
                    │  MpcController   → U2, U3, U4 (torques)   │
                    │  mixRotors       → rotor speeds           │
                    │  rk4Step         → true nonlinear plant   │
                    └──────────────────────────────────────────┘
```

A quadrotor can only produce thrust along its body z-axis — it moves horizontally by tilting. The outer loop converts "where I need to be" into "how I should tilt"; the inner loop makes that tilt happen. The two loops run at different rates because attitude dynamics are fast (milliseconds) and position dynamics are slow (seconds).

## Features

**Control**
- Outer-loop position control via state feedback linearisation with pole-placed gains
- Inner-loop attitude control via linear-parameter-varying MPC — the rotational model is re-linearised around the measured state every 100 ms
- Condensed QP solved in input increments (velocity form), which gives integral action structurally rather than through an added integrator
- Solved with `ldlt()` rather than an explicit matrix inverse

**Estimation**
- Three independent scalar RLS filters identifying Ix, Iy, Iz online from the rotational equations of motion, with an exponential forgetting factor
- Perturbation framework: the plant's true physics and the controller's believed model are separate C++ types, so the simulator can fly with a deliberate parameter mismatch and measure exactly what adaptation buys

**Design**
- C++20 concepts (`TrajectoryGenerator`, `PlantDynamics`, `InertiaEstimator`) as contracts, verified by `static_assert` at compile time
- `std::variant` + `std::visit` for runtime trajectory selection — no vtables, no heap
- `Simulation<Plant, Estimator>` template policies; `NoEstimator` has an empty `update()` that the compiler removes entirely, so the non-adaptive build pays nothing
- Fixed-size Eigen types throughout the hot loop — stack allocated, dimensions checked at compile time
- Concurrent comparison runs via `std::async`, lock-free by construction (each simulation owns its mutable state; shared reference data is const)

## Requirements

- C++20 compiler (MSVC 2022, GCC 11+, or Clang 14+)
- Eigen 3.4+

Header-only apart from `main.cpp` — templates live in headers anyway, and it drops into a Visual Studio project with no build configuration.

## Building

**Visual Studio:** add the files to an empty C++ project, set the language standard to C++20, and point *Additional Include Directories* at `include/` and your Eigen path (or `vcpkg install eigen3`).

**Command line:**

```bash
g++ -std=c++20 -O2 -Wall -Wextra -I include -I /path/to/eigen src/main.cpp -o drone_sim
./drone_sim
```

## Output

```
Trajectory 1, 251 outer steps, 100.0000 s

Nominal          RMS = 0.5285 m
Perturbed fixed  RMS = 0.5296 m
Perturbed RLS    RMS = 0.5302 m  (-0.1295% improvement)

Believed inertia at end of flight (true in brackets):
  Ix = 0.0044  (0.0044)
  Iy = 0.0044  (0.0044)
  Iz = 0.0048  (0.0048)
```

Three CSVs are written (`nominal.csv`, `perturbed_fixed.csv`, `perturbed_rls.csv`) containing the full state, input, believed-inertia, and interpolated-reference history at every inner step — ready for plotting.

## Results

Nominal tracking RMS across the four reference trajectories:

| Trajectory | RMS [m] |
|---|---|
| 1 — Helix (2.5 laps, 5 → 25 m climb) | 0.5285 |
| 2 — Straight line | 0.4398 |
| 3 — Wave | 0.3873 |
| 4 — Vertical wave (helix + bobbing) | 0.6940 |

RMS is computed over the whole flight including the initial transient (the drone starts 1 m off-path); steady-state tracking is around 0.15 m.

### The adaptation experiment

The headline experiment flies the drone with a plant 30% heavier in roll and pitch and 20% lighter in yaw than the controller believes, comparing a fixed-model controller against one using RLS.

**The estimator converges exactly** — the believed inertias land on the true perturbed values in every run. **Tracking error barely moves** (~0.1%, sub-millimetre).

That result is the interesting one, and it is the correct outcome for this architecture. Two mechanisms in the baseline already absorb a 30% parametric error: the LPV model is re-linearised around the measured state ten times a second, so it is continuously re-fit to observed behaviour, and the input-increment MPC formulation carries integral action, so persistent under-response is met by accumulated torque. The controller compensates for the mismatch without ever knowing what the mismatch is.

The estimator's value therefore shows up in the convergence trace rather than the tracking score — and would show up in tracking under larger mismatch, faster manoeuvres where a 100 ms-stale model matters, or in a controller without per-step re-linearisation.

A secondary result visible across the trajectories: the magnitude of the RLS effect shrinks monotonically with how much the trajectory excites the rotational dynamics (helix −0.13% → wave −0.004%). That is persistence of excitation — when the drone barely rotates, the regressors go to zero, the filter gain goes to zero, and the estimator correctly learns nothing. On the gentle trajectories the estimates converge almost entirely during the initial recovery transient, which is the only aggressive manoeuvring in the flight.

## Verification

The build was verified against an independently developed reference implementation, matching to four decimal places on all four trajectories (trajectory 4 differs by 0.2 mm — floating point ordering between compilers). Matching an independent implementation to sub-millimetre RMS over 1001 coupled nonlinear steps jointly certifies the equations of motion, integrator, both controllers, mixer, and trajectory generation.

Two bugs were found during that verification, and both were diagnosed from the output before opening the source:

**Free fall.** All three runs reported identical RMS of 21,967 m while the RLS still converged perfectly. Perfect convergence meant the rotational path was healthy; identical results across runs meant the fault was inertia-independent. The CSV settled it: `w = -g·t` and `z = -½gt²` exactly — free fall — while thrust was being commanded. Thrust commanded but not acting isolated the `+U1/m` term in the ẇ equation, and the input was being unpacked from index 1 (the roll torque) instead of index 0. Even the residual 0.0009 in the first velocity sample was the roll torque leaking in as thrust.

**A non-orthogonal rotation matrix.** The pitch rotation's third row was `(0, 0, 1)` instead of `(-sinθ, 0, cosθ)`, giving `det(Ry) = cosθ` — not a rotation. Masked by small pitch angles on these trajectories, it would corrupt the body-to-inertial transform under any aggressive manoeuvre.

Both bugs are now covered by unit tests: a hover-equilibrium assertion (level attitude, `U1 = mg` ⇒ all derivatives zero) catches the first, and an orthogonality check (`RᵀR = I`, `det R = 1`) catches the second.

## Project structure

```
include/
  Config.hpp              constexpr constants, tuning parameters
  Trajectory.hpp          four trajectories, concept, variant dispatch, yaw unwrapping
  DroneModel.hpp          plant concept, equations of motion, RK4, LPV model, rotor mixer
  PositionController.hpp  outer loop
  Mpc.hpp                 inner loop — condensed QP
  Rls.hpp                 estimator concept, NoEstimator, RlsEstimator
  Simulation.hpp          templated orchestrator
  Csv.hpp                 output and error metrics
src/
  main.cpp                nominal run plus concurrent perturbed comparison
tests/
  GoogleTest suite
```

## Known limitations

Each is deliberate, and each has a scoped fix:

- **Euler angles** — singular at ±90° pitch. Unreachable on these trajectories; a quaternion plant would slot in behind the existing `PlantDynamics` concept, though the MPC would need reformulating around an error quaternion.
- **Unconstrained MPC** — the rotor mixer throws on an infeasible demand rather than the optimiser clamping it. A constrained QP with rotor box limits is the fix.
- **Finite-difference accelerations in the RLS** — clean in simulation; on hardware this amplifies gyro noise and would need filtering or a model-based estimate.
- **Forward-Euler discretisation of the LPV model** — sound at the current sample time, degrades if it grows.
- **No drag or motor lag** — each is a single new struct satisfying `PlantDynamics`.
- **Mass held fixed** — deliberately reserved for a Kalman filter, since payload changes have the kind of evolution structure a Kalman filter models well and RLS does not.

## Roadmap

- Kalman filter for online mass estimation, as a second estimator policy
- Input-constrained MPC (rotor limits)
- Larger-mismatch perturbation studies, where adaptation should measurably improve tracking
- Model reference adaptive control as an augmentation on the MPC baseline
- Quaternion attitude representation

## References

- Maciejowski, *Predictive Control with Constraints* — condensed MPC formulation
- Åström & Wittenmark, *Adaptive Control* — RLS, forgetting factors, persistence of excitation
- Beard & McLain, *Small Unmanned Aircraft* — quadrotor equations of motion, cascaded control
