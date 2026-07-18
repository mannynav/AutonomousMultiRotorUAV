
#pragma once
#include <Eigen/Dense>
#include "Config.hpp"
#include "Trajectories.hpp"
#include "Dronemodel.hpp"
#include "PositionController.hpp"
#include "Mpc.hpp"
#include "RLS.hpp"

// ============================================================
//  Simulation.
//
//  Templated on both the plant and the estimator policy:
//
//    Simulation<RigidBodyPlant> in Dronemodel.hpp               nominal, no adaptation
//    Simulation<RigidBodyPlant, RlsEstimator> in RLS.hpp		 adaptive
//
//  The plant carries the TRUE physics. The believed inertia is
//  seeded from config and after that only the estimator touches
//  it - config stays immutable.
//
//  Cascade per outer step (every Ts * inner_loop_length):
//    position controller -> phi_ref, theta_ref, U1
//    then inner_loop_length times (every Ts):
//      estimator -> LPV(believed) -> MPC -> mixer -> RK4(plant)
// ============================================================

struct SimResult {
	Eigen::ArrayXd time;		// inner loop time axis
	Eigen::MatrixXd states;		// (N, 12) one row per inner step
	Eigen::MatrixXd inputs;		// (N, 4)
	Eigen::MatrixXd believed;	// (N, 3) inertia estimate trace
};

template<PlantDynamics Plant, InertiaEstimator Estimator = NoEstimator>
class Simulation {
public:
	Simulation(Config cfg, Plant plant, Estimator estimator = {})
		: cfg_{ std::move(cfg) }
		, plant_{ std::move(plant) }
		, estimator_{ std::move(estimator) }
		, pos_ctrl_{ cfg_.control, cfg_.physical.mass }
		, mpc_{ cfg_.control } {
	}

	SimResult run(const TrajectoryData& ref, const Eigen::ArrayXd& psi_ref,
		const Eigen::ArrayXd& t_outer) const {
		const auto& ctrl = cfg_.control;
		const int inner = ctrl.inner_loop_length;
		const double Ts = ctrl.Ts;
		const Eigen::Index n_outer = t_outer.size();
		const Eigen::Index n_steps = (n_outer - 1) * inner + 1;

		// Believed inertia: seeded from config, owned by the estimator
		// from here on. The plant's true params are never consulted.
		InertiaEstimate believed{ cfg_.physical.Ix, cfg_.physical.Iy, cfg_.physical.Iz };
		Estimator estimator = estimator_;	// estimators carry state, run on a copy

		// Initial condition matches the reference start
		State s = State::Zero();
		s(7) = -1.0;			// y
		s(11) = psi_ref(0);		// psi

		// Hover-ish initial rotor speeds -> initial inputs
		const double w0 = 110.0 * constants::pi / 3.0;
		Input u;
		u(0) = constants::ct * 4.0 * w0 * w0;
		u(1) = 0.0;
		u(2) = 0.0;
		u(3) = 0.0;
		double omega_total = 0.0;	// symmetric rotors cancel

		// Preallocated histories - one write per step, no reallocation
		SimResult res;
		res.time = Eigen::ArrayXd::LinSpaced(n_steps, 0.0, (n_steps - 1) * Ts);
		res.states.resize(n_steps, 12);
		res.inputs.resize(n_steps, 4);
		res.believed.resize(n_steps, 3);

		auto log = [&](Eigen::Index row) {
			res.states.row(row) = s.transpose();
			res.inputs.row(row) = u.transpose();
			res.believed.row(row) << believed.Ix, believed.Iy, believed.Iz;
			};
		log(0);
		Eigen::Index row = 1;

		for (Eigen::Index i_g = 0; i_g + 1 < n_outer; ++i_g) {

			// ---- Outer loop: position controller -----------------------
			const ReferencePoint rp{
				ref.x(i_g + 1), ref.x_dot(i_g + 1), ref.x_ddot(i_g + 1),
				ref.y(i_g + 1), ref.y_dot(i_g + 1), ref.y_ddot(i_g + 1),
				ref.z(i_g + 1), ref.z_dot(i_g + 1), ref.z_ddot(i_g + 1),
				psi_ref(i_g + 1)
			};
			const AttitudeCommand cmd = pos_ctrl_.compute(rp, s);
			u(0) = cmd.U1;

			// Reference signals for the MPC: phi/theta held constant over
			// the inner window, psi interpolated linearly between outer steps
			const int n_ref = inner + 1;
			Eigen::VectorXd ref_signals(3 * n_ref);
			for (int k = 0; k < n_ref; ++k) {
				const double frac = static_cast<double>(k) / inner;
				ref_signals(3 * k) = cmd.phi_ref;
				ref_signals(3 * k + 1) = cmd.theta_ref;
				ref_signals(3 * k + 2) = psi_ref(i_g)
					+ (psi_ref(i_g + 1) - psi_ref(i_g)) * frac;
			}

			// ---- Inner loop: estimate -> linearise -> optimise -> step --
			int hz = ctrl.horizon;
			int k_ptr = 0;
			for (int i = 0; i < inner; ++i) {

				// Estimator sees the state/input pair from before this
				// step's MPC acts - matches the finite difference timing
				estimator.update(s, u, omega_total, believed);

				const LpvMatrices lpv = lpvDiscrete(believed, s, omega_total, Ts);

				// Augmented state for the incremental MPC formulation
				const Eigen::Vector3d rates = eulerRateMatrix(s(9), s(10)) * s.segment<3>(3);
				Eigen::Matrix<double, 9, 1> x_aug;
				x_aug << s(9), rates(0), s(10), rates(1), s(11), rates(2),
					u(1), u(2), u(3);

				// Sliding reference window - horizon shrinks at the end
				k_ptr += 3;
				Eigen::VectorXd window;
				if (k_ptr + 3 * hz <= ref_signals.size()) {
					window = ref_signals.segment(k_ptr, 3 * hz);
				}
				else {
					window = ref_signals.segment(k_ptr, ref_signals.size() - k_ptr);
					hz = static_cast<int>(window.size()) / 3;
				}

				const Eigen::Vector3d du = mpc_.solve(lpv, x_aug, window, hz);
				u(1) += du(0);
				u(2) += du(1);
				u(3) += du(2);

				// Torque demands -> rotor speeds -> gyroscopic term
				omega_total = mixRotors(u).omega_total;

				// Step the TRUE plant
				s = rk4Step(plant_, s, omega_total, u, Ts);

				log(row++);
			}
		}
		return res;
	}

private:
	Config cfg_;
	Plant plant_;
	Estimator estimator_;
	PositionController pos_ctrl_;
	MpcController mpc_;
};