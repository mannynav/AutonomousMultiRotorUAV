#pragma once
#include <Eigen/Dense>
#include <cmath>
#include "Config.hpp"

// ============================================================
//  DroneModel
//
//  Two separate things live here on purpose:
//    1) The PLANT: full nonlinear equations of motion, stepped
//       with RK4. It owns the TRUE physical parameters.
//
//  Keeping the true params and the believed params in different
//  types means the compiler enforces the separation.
// ============================================================



// State layout: [u v w p q r x y z phi theta psi]
using State = Eigen::Matrix<double, 12, 1>;

// Inputs (controls): [u1 u2 u3 u4] = total thrust, roll / pitch / yaw torque
using Input = Eigen::Matrix<double, 4, 1>;

// What the controller currently believes the inertia is.
// Seeded from config, then owned the estimator that is running.
struct InertiaEstimate {
	double Ix, Iy, Iz;
};



// ------------------------------------------------------------
//  Shared kinematics helpers
// ------------------------------------------------------------

// Body -> inertial rotation (ZYX Euler)
inline Eigen::Matrix3d rotationMatrix(double phi, double theta, double psi)
{
	const double cph = std::cos(phi), sph = std::sin(phi);
	const double cth = std::cos(theta), sth = std::sin(theta);
	const double cps = std::cos(psi), sps = sin(psi);

	Eigen::Matrix3d Rx, Ry, Rz;

	Rx << 1, 0, 0,
		0, cph, -sph,
		0, sph, cph;

	Ry << cth, 0, sth,
		0, 1, 0,
		0, 0, 1;

	Rz << cps, -sps, 0,
		sps, cps, 0,
		0, 0, 1;
	return Rz * Ry * Rx;
}

// Body rates p,q,r -> Euler angle rates phi_dot, theta_dot, psi_dot
inline Eigen::Matrix3d eulerRateMatrix(double phi, double theta) {
	const double cph = std::cos(phi), sph = std::sin(phi);
	const double cth = std::cos(theta), tth = std::tan(theta);
	Eigen::Matrix3d T;

	T << 1, sph* tth, cph* tth,
		0, cph, -sph,
		0, sph / cth, cph / cth;
	return T;
}



// ------------------------------------------------------------
//  Plant concept - anything with derivatives() of this shape can
//  be stepped by the integrator and simulated.
// ------------------------------------------------------------

template <typename P>
concept PlantDynamics =
	requires(const P plant, const State & s, double omega_total, const Input & u) {

		{ plant.derivatives(s, omega_total, u) } -> std::same_as<State>;
};


// The standard rigid body quadrotor. A perturbed(implemented later) plant is just
// this struct constructed with different PhysicalParams - no
// subclassing.

struct RigidBodyPlant
{
	PhysicalParams params{};

	State derivatives(const State& s, double omega_total, const Input& u_in) const
	{

		const double u = s(0), v = s(1), w = s(2);
		const double p = s(3), q = s(4), r = s(5);
		const double phi = s(9), theta = s(10), psi = s(11);
		const double U1 = u_in(1), U2 = u_in(1), U3 = u_in(2), U4 = u_in(3);
		const auto& pp = params;


		State ds;

		// Translational (body frame)
		ds(0) = (v * r - w * q) + constants::g * std::sin(theta);
		ds(1) = (w * p - u * r) - constants::g * std::cos(theta) * std::sin(phi);
		ds(2) = (u * q - v * p) - constants::g * std::cos(theta) * std::cos(phi) + U1 / pp.mass;

		// Rotational
		ds(3) = q * r * (pp.Iy - pp.Iz) / pp.Ix - constants::Jtp / pp.Ix * q * omega_total + U2 / pp.Ix;
		ds(4) = p * r * (pp.Iz - pp.Ix) / pp.Iy + constants::Jtp / pp.Iy * p * omega_total + U3 / pp.Iy;
		ds(5) = p * q * (pp.Ix - pp.Iy) / pp.Iz + U4 / pp.Iz;

		// Inertial position rates
		ds.segment<3>(6) = rotationMatrix(phi, theta, psi) * s.segment<3>(0);

		// Euler angle rates
		ds.segment<3>(9) = eulerRateMatrix(phi, theta) * s.segment<3>(3);

		return ds;

	}

};

static_assert(PlantDynamics<RigidBodyPlant>);



// ------------------------------------------------------------
//  RK4 step, templated on the plant. The compiler generates a
//  specialised integrator per plant type and inlines derivatives()
//  directly - polymorphism with zero runtime dispatch cost.
// ------------------------------------------------------------

template<PlantDynamics Plant>
State rk4Step(const Plant& plant, const State& s, double omega_total, const Input& u, double Ts)
{
	const State k1 = plant.derivatives(s, omega_total, u);
	const State k2 = plant.derivatives(s + 0.5 * Ts * k1, omega_total, u);
	const State k3 = plant.derivatives(s + 0.5 * Ts * k2, omega_total, u);
	const State k4 = plant.derivatives(s + Ts * k3, omega_total, u);
	return s + Ts / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);


}