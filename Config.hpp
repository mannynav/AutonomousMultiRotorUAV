
#pragma once
#include <Eigen/Dense>
#include <complex>
#include <numbers>

// ============================================================
//  Config - all constants and tuning knobs live here.
//  True physical constants are constexpr.
//  Tuning parameters are regular members.
// ============================================================

namespace constants {
	inline constexpr double g = 9.81;
	inline constexpr double pi = std::numbers::pi;
	inline constexpr double two_pi = 2.0 * pi;

	// Rotor coefficients (Astec Hummingbird). The datasheet values are
	// per-RPM so convert to rad/s once here instead of everywhere else.
	inline constexpr double rpm_conv = 60.0 / two_pi;
	inline constexpr double ct = 7.6184e-8 * rpm_conv * rpm_conv;	// thrust coeff [N*s^2]
	inline constexpr double cq = 2.6839e-9 * rpm_conv * rpm_conv;	// torque coeff [N*m*s^2]
	inline constexpr double arm_length = 0.171;						// [m]
	inline constexpr double Jtp = 1.302e-6;							// rotor inertia [kg*m^2]
}


// Inertia + mass of the airframe. The plant owns the true values.
// The controller only ever gets a believed copy seeded from here -
// after that the estimator owns the belief, not this struct.

struct PhysicalParams {
	double Ix{ 0.0034 };	// [kg*m^2]
	double Iy{ 0.0034 };	// [kg*m^2]
	double Iz{ 0.006 };		// [kg*m^2]
	double mass{ 0.698 };	// [kg]
};

struct TrajectoryParams {
	double helix_radius{ 2.0 };			// [m]
	double helix_frequency{ 0.025 };	// [Hz]
	double initial_height{ 5.0 };		// [m]
	double final_height{ 25.0 };		// [m]
};

struct ControlParams {
	double Ts{ 0.1 };				// inner loop sample time [s]
	int horizon{ 4 };				// MPC horizon length
	int inner_loop_length{ 4 };		// inner iterations per outer step

	// Cost weights - stored as diagonals only so a non-diagonal
	// weight matrix is impossible to construct by accident.
	Eigen::Vector3d Q{ 10.0, 10.0, 10.0 };	// output weights
	Eigen::Vector3d S{ 20.0, 20.0, 20.0 };	// final horizon weights
	Eigen::Vector3d R{ 10.0, 10.0, 10.0 };	// input weights

	// Position controller poles. Stored as one complex value per axis,
	// the conjugate is implied (gain formulas below assume the pair).
	std::complex<double> pole_x{ -0.5, 0.3 };
	std::complex<double> pole_y{ -0.5, 0.3 };
	std::complex<double> pole_z{ -1.0, 1.3 };
};

struct SimParams {
	double total_time{ 100.0 };		// [s]
	int trajectory_id{ 1 };			// 1=Helix 2=StraightLine 3=Wave 4=VerticalWave
};

struct Config {
	PhysicalParams physical{};
	TrajectoryParams trajectory{};
	ControlParams control{};
	SimParams sim{};
};