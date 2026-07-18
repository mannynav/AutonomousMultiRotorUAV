
#pragma once
#include <Eigen/Dense>
#include <numbers>
#include <variant>
#include <stdexcept>
#include "Config.hpp"

// ============================================================
//  This header will hold trajectories for the drone to follow.
//
//  Each trajectory is a plain struct with one const method that
//  returns the full TrajectoryData bundle. There is no base class -
//  the TrajectoryGenerator concept below is the contract, and
//  std::variant + std::visit handles runtime selection without
//  vtables or heap allocation.
// ============================================================


// Data bundle that every trajectory will return
struct TrajectoryData {
	Eigen::ArrayXd x, y, z;
	Eigen::ArrayXd x_dot, y_dot, z_dot;
	Eigen::ArrayXd x_ddot, y_ddot, z_ddot;
};

// ------------------------------------------------------------
//  The contract. A type is a trajectory generator if calling
//  generateTrajectory on a const instance with these arguments
//  compiles and returns TrajectoryData. No inheritance needed -
//  any struct with the right shape qualifies automatically.
// ------------------------------------------------------------
template<typename T>
concept TrajectoryGenerator =
	requires(const T gen, const TrajectoryParams & params,
const Eigen::ArrayXd & time_vector, double delta_height) {
	{ gen.generateTrajectory(params, time_vector, delta_height) }
	-> std::same_as<TrajectoryData>;
};

// ------------------------------------------------------------
//  Trajectory 1 - circular helix climbing at constant rate
// ------------------------------------------------------------
struct Helix {
	TrajectoryData generateTrajectory(const TrajectoryParams& params, const Eigen::ArrayXd& time_vector, double delta_height) const {
		const Eigen::Index t_length = time_vector.size();
		const double final_t = time_vector(Eigen::last);
		const double frequency = constants::two_pi * params.helix_frequency;
		const Eigen::ArrayXd alpha = frequency * time_vector;
		const double r = params.helix_radius;
		TrajectoryData out;
		// Position
		out.x = r * alpha.cos();
		out.y = r * alpha.sin();
		out.z = params.initial_height + delta_height / final_t * time_vector;
		// Velocity
		out.x_dot = -r * alpha.sin() * frequency;
		out.y_dot = r * alpha.cos() * frequency;
		out.z_dot = Eigen::ArrayXd::Constant(t_length, delta_height / final_t);
		// Acceleration
		out.x_ddot = -r * alpha.cos() * frequency * frequency;
		out.y_ddot = -r * alpha.sin() * frequency * frequency;
		out.z_ddot = Eigen::ArrayXd::Zero(t_length);
		return out;
	}
};

// ------------------------------------------------------------
//  Trajectory 2 - straight diagonal line, constant climb
// ------------------------------------------------------------
struct StraightLine {
	TrajectoryData generateTrajectory(const TrajectoryParams& params, const Eigen::ArrayXd& time_vector, double delta_height) const {
		const Eigen::Index t_length = time_vector.size();
		const double final_t = time_vector(Eigen::last);
		TrajectoryData out;
		// Position
		out.x = 2.0 * time_vector / 20.0 + 1.0;
		out.y = 2.0 * time_vector / 20.0 - 2.0;
		out.z = params.initial_height + delta_height / final_t * time_vector;

		// Velocity
		out.x_dot = Eigen::ArrayXd::Constant(t_length, 0.1);
		out.y_dot = Eigen::ArrayXd::Constant(t_length, 0.1);
		out.z_dot = Eigen::ArrayXd::Constant(t_length, delta_height / final_t);

		// Acceleration
		out.x_ddot = Eigen::ArrayXd::Zero(t_length);
		out.y_ddot = Eigen::ArrayXd::Zero(t_length);
		out.z_ddot = Eigen::ArrayXd::Zero(t_length);
		return out;
	}
};

// ------------------------------------------------------------
//  Trajectory 3 - sideways sine wave drifting in x/y
// ------------------------------------------------------------
struct Wave {
	TrajectoryData generateTrajectory(const TrajectoryParams& params, const Eigen::ArrayXd& time_vector, double delta_height) const {
		const Eigen::Index t_length = time_vector.size();
		const double final_t = time_vector(Eigen::last);
		const double frequency = constants::two_pi * params.helix_frequency;
		const Eigen::ArrayXd alpha = frequency * time_vector;
		const double r = params.helix_radius;
		TrajectoryData out;
		// Position
		out.x = r / 5.0 * alpha.sin() + time_vector / 100.0;
		out.y = time_vector / 100.0 - 1.0;
		out.z = params.initial_height + delta_height / final_t * time_vector;
		// Velocity
		out.x_dot = r / 5.0 * alpha.cos() * frequency + 0.01;
		out.y_dot = Eigen::ArrayXd::Constant(t_length, 0.01);
		out.z_dot = Eigen::ArrayXd::Constant(t_length, delta_height / final_t);
		// Acceleration
		out.x_ddot = -r / 5.0 * alpha.sin() * frequency * frequency;
		out.y_ddot = Eigen::ArrayXd::Zero(t_length);
		out.z_ddot = Eigen::ArrayXd::Zero(t_length);
		return out;
	}
};

// ------------------------------------------------------------
//  Trajectory 4 - helix in x/y with the z axis bobbing up
//  and down. ww scales how fast the vertical bob oscillates.
// ------------------------------------------------------------
struct VerticalWave {
	TrajectoryData generateTrajectory(const TrajectoryParams& params, const Eigen::ArrayXd& time_vector, double delta_height) const {
		const double final_t = time_vector(Eigen::last);
		const double ww = 1.0;	// vertical wave angular frequency
		const double frequency = constants::two_pi * params.helix_frequency;
		const Eigen::ArrayXd alpha = frequency * time_vector;
		const double r = params.helix_radius;
		TrajectoryData out;
		// Position
		out.x = r * alpha.cos();
		out.y = r * alpha.sin();
		out.z = params.initial_height + 7.0 * delta_height / final_t * (ww * time_vector).sin();
		// Velocity
		out.x_dot = -r * alpha.sin() * frequency;
		out.y_dot = r * alpha.cos() * frequency;
		out.z_dot = 7.0 * delta_height / final_t * (ww * time_vector).cos() * ww;
		// Acceleration
		out.x_ddot = -r * alpha.cos() * frequency * frequency;
		out.y_ddot = -r * alpha.sin() * frequency * frequency;
		out.z_ddot = -7.0 * delta_height / final_t * (ww * time_vector).sin() * ww * ww;
		return out;
	}
};

// Sanity check the contract at compile time - if one of the structs
// above drifts out of shape, the error shows up here.
static_assert(TrajectoryGenerator<Helix>);
static_assert(TrajectoryGenerator<StraightLine>);
static_assert(TrajectoryGenerator<Wave>);
static_assert(TrajectoryGenerator<VerticalWave>);

// ------------------------------------------------------------
//  Runtime selection. The trajectory id comes from config, so
//  the concrete type isn't known at compile time. std::variant
//  is the bridge: closed set of types, value semantics, no
//  vtables. std::visit dispatches to the right generateTrajectory.
// ------------------------------------------------------------
using AnyTrajectory = std::variant<Helix, StraightLine, Wave, VerticalWave>;

inline AnyTrajectory makeTrajectory(int id) {
	switch (id) {
	case 1: return Helix{};
	case 2: return StraightLine{};
	case 3: return Wave{};
	case 4: return VerticalWave{};
	default: throw std::invalid_argument("trajectory_id must be 1-4");
	}
}

inline TrajectoryData generate(const AnyTrajectory& trajectory, const TrajectoryParams& params,
	const Eigen::ArrayXd& time_vector, double delta_height) {

	// The constrained auto parameter means the lambda only accepts types
	// satisfying the concept - visit over the variant stays type safe.
	return std::visit(
		[&](const TrajectoryGenerator auto& gen) {
			return gen.generateTrajectory(params, time_vector, delta_height);
		},
		trajectory);
}

// ------------------------------------------------------------
//  Yaw reference. This one is stateful - each step depends on
//  the previous one because of the +-pi wraparound correction,
//  so it genuinely needs a loop. Everything else vectorizes.
// ------------------------------------------------------------
inline Eigen::ArrayXd computeYawReference(const TrajectoryData& traj) {
	const Eigen::Index n = traj.x.size();

	// Per-sample position deltas, first element repeated like np.diff + prepend
	Eigen::ArrayXd dx(n), dy(n);
	dx.tail(n - 1) = traj.x.tail(n - 1) - traj.x.head(n - 1);
	dy.tail(n - 1) = traj.y.tail(n - 1) - traj.y.head(n - 1);
	dx(0) = dx(1);
	dy(0) = dy(1);

	Eigen::ArrayXd psi(n);
	psi(0) = std::atan2(traj.y(0), traj.x(0)) + constants::pi / 2.0;
	for (Eigen::Index i = 1; i < n; ++i)
		psi(i) = std::atan2(dy(i), dx(i));

	// Unwrap so the reference tracks total rotation instead of
	// jumping when atan2 crosses +-pi
	Eigen::ArrayXd psi_int(n);
	psi_int(0) = psi(0);
	for (Eigen::Index i = 1; i < n; ++i) {
		double dpsi = psi(i) - psi(i - 1);
		if (dpsi < -constants::pi)      dpsi += constants::two_pi;
		else if (dpsi > constants::pi)  dpsi -= constants::two_pi;
		psi_int(i) = psi_int(i - 1) + dpsi;
	}
	return psi_int;
}

// Time vector for the outer loop. LinSpaced wants a count, not a
// step, so compute the count from the step first.
inline Eigen::ArrayXd makeTimeVector(const Config& cfg) {
	const double step = cfg.control.Ts * cfg.control.inner_loop_length;
	const int count = static_cast<int>(std::round(cfg.sim.total_time / step)) + 1;
	return Eigen::ArrayXd::LinSpaced(count, 0.0, cfg.sim.total_time);
}