
#pragma once
#include <Eigen/Dense>
#include <array>
#include <optional>
#include <vector>
#include "Config.hpp"
#include "DroneModel.hpp"

// ============================================================
//  Rls - online inertia estimation.
//
//  Three independent scalar RLS filters, one per axis, derived
//  from the rotational equations of motion:
//
//    Ix:  y = U2 + (Iy-Iz) q r - Jtp q W      phi_reg = p_dot
//    Iy:  y = U3 + (Iz-Ix) p r + Jtp p W      phi_reg = q_dot
//    Iz:  y = U4 + (Ix-Iy) p q                phi_reg = r_dot
//
//  Angular accelerations come from finite differences - fine in
//  sim where states are clean, would need filtering on hardware.
//
//  Mass is deliberately NOT estimated here. Might get its own Kalman filter later
//
//  The estimator is a POLICY: Simulation is templated on the
//  estimator type, and NoEstimator below is the zero-cost null
//  policy for running without adaptation. The concept defines
//  the contract both must satisfy.
// ============================================================

template<typename E>
concept InertiaEstimator =
	requires(E est, const State & s, const Input & u, double omega_total, InertiaEstimate & believed) {
		{ est.update(s, u, omega_total, believed) } -> std::same_as<void>;
};


// Null policy - no adaptation, belief stays at the seed values.
// update() is empty so the compiler removes the call entirely.
struct NoEstimator {
	void update(const State&, const Input&, double, InertiaEstimate&) {}
};

class RlsEstimator {
public:
	RlsEstimator(double Ts, double forgetting_factor = 0.98, double P0 = 1e4)
		: Ts_{ Ts }, lambda_{ forgetting_factor } {
		P_.fill(P0);
	}

	void update(const State& s, const Input& u, double omega_total, InertiaEstimate& believed) {
		if (!prev_state_) {
			prev_state_ = s;	// first call just caches, nothing to difference yet
			return;
		}

		// Finite difference angular accelerations
		const double p_dot = (s(3) - (*prev_state_)(3)) / Ts_;
		const double q_dot = (s(4) - (*prev_state_)(4)) / Ts_;
		const double r_dot = (s(5) - (*prev_state_)(5)) / Ts_;
		const double p = s(3), q = s(4), r = s(5);

		// Cross coupling terms use the current best estimates
		const std::array<double, 3> regressors{ p_dot, q_dot, r_dot };
		const std::array<double, 3> measurements{
			u(1) + (believed.Iy - believed.Iz) * q * r - constants::Jtp * q * omega_total,
			u(2) + (believed.Iz - believed.Ix) * p * r + constants::Jtp * p * omega_total,
			u(3) + (believed.Ix - believed.Iy) * p * q
		};

		std::array<double*, 3> targets{ &believed.Ix, &believed.Iy, &believed.Iz };

		for (int i = 0; i < 3; ++i) {
			const double phi = regressors[i];
			const double e = measurements[i] - phi * (*targets[i]);
			const double K = P_[i] * phi / (lambda_ + phi * phi * P_[i]);
			*targets[i] += K * e;
			P_[i] = (1.0 / lambda_) * (1.0 - K * phi) * P_[i];
			// Never let an estimate go nonphysical
			*targets[i] = std::max(*targets[i], 1e-6);
		}

		prev_state_ = s;
	}

	const std::array<double, 3>& covariance() const { return P_; }

private:
	double Ts_;
	double lambda_;
	std::array<double, 3> P_{};
	std::optional<State> prev_state_{};	// empty until the first update
};

static_assert(InertiaEstimator<NoEstimator>);
static_assert(InertiaEstimator<RlsEstimator>);