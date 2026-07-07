
#pragma once
#include <Eigen/Dense>
#include <complex>
#include <cmath>
#include "Config.hpp"
#include "DroneModel.hpp"

// ============================================================
//  PositionController - the outer loop.
//
//  State feedback linearisation: given where the drone should be,
//  compute what attitude (phi, theta) and total thrust (U1) will
//  get it there. Gains come from pole placement and are computed
//  once in the constructor.
// ============================================================

// Single point of the reference the controller tracks this step
struct ReferencePoint {
	double x, x_dot, x_ddot;
	double y, y_dot, y_ddot;
	double z, z_dot, z_ddot;
	double psi;
};

struct AttitudeCommand {
	double phi_ref;
	double theta_ref;
	double U1;
};

class PositionController {
public:
	PositionController(const ControlParams& ctrl, double mass)
		: mass_{ mass } {

		// For a conjugate pole pair a +- bi the feedback gains reduce to
		//   k1 = -(a^2 + b^2) = -|p|^2      k2 = 2a
		// which is what the general pole placement formula collapses to.
		kx1_ = -std::norm(ctrl.pole_x);  kx2_ = 2.0 * ctrl.pole_x.real();
		ky1_ = -std::norm(ctrl.pole_y);  ky2_ = 2.0 * ctrl.pole_y.real();
		kz1_ = -std::norm(ctrl.pole_z);  kz2_ = 2.0 * ctrl.pole_z.real();
	}

	AttitudeCommand compute(const ReferencePoint& ref, const State& s) const {
		const double phi = s(9), theta = s(10), psi = s(11);

		// Body velocity -> inertial velocity
		const Eigen::Vector3d vel = rotationMatrix(phi, theta, psi) * s.segment<3>(0);

		// Tracking errors
		const double ex = ref.x - s(6), ex_dot = ref.x_dot - vel(0);
		const double ey = ref.y - s(7), ey_dot = ref.y_dot - vel(1);
		const double ez = ref.z - s(8), ez_dot = ref.z_dot - vel(2);

		// Desired accelerations from pole-placed feedback + feedforward
		const double vx = ref.x_ddot - (kx1_ * ex + kx2_ * ex_dot);
		const double vy = ref.y_ddot - (ky1_ * ey + ky2_ * ey_dot);
		const double vz = ref.z_ddot - (kz1_ * ez + kz2_ * ez_dot);

		// Invert the thrust-vector geometry to get the attitude that
		// produces those accelerations at the reference yaw
		const double a = vx / (vz + constants::g);
		const double b = vy / (vz + constants::g);
		const double c = std::cos(ref.psi);
		const double d = std::sin(ref.psi);

		const double theta_ref = std::atan(a * c + b * d);

		// The phi formula divides by cos(psi) or sin(psi) depending on
		// which quadrant psi sits in - pick the branch that keeps the
		// denominator away from zero.
		double psi_s = (ref.psi >= 0.0)
			? ref.psi - std::floor(std::abs(ref.psi) / constants::two_pi) * constants::two_pi
			: ref.psi + std::floor(std::abs(ref.psi) / constants::two_pi) * constants::two_pi;
		const double abs_psi = std::abs(psi_s);

		double tan_phi;
		if ((abs_psi < constants::pi / 4.0 || abs_psi > 7.0 * constants::pi / 4.0) ||
			(abs_psi > 3.0 * constants::pi / 4.0 && abs_psi < 5.0 * constants::pi / 4.0))
			tan_phi = std::cos(theta_ref) * (std::tan(theta_ref) * d - b) / c;
		else
			tan_phi = std::cos(theta_ref) * (a - std::tan(theta_ref) * c) / d;

		const double phi_ref = std::atan(tan_phi);
		const double U1 = (vz + constants::g) * mass_
			/ (std::cos(phi_ref) * std::cos(theta_ref));

		return { phi_ref, theta_ref, U1 };
	}

private:
	double mass_;
	double kx1_, kx2_, ky1_, ky2_, kz1_, kz2_;
};