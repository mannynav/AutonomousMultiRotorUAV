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
		const double U1 = u_in(0), U2 = u_in(1), U3 = u_in(2), U4 = u_in(3);
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


// ------------------------------------------------------------
//  LPV linearisation of the rotational dynamics, discretised
//  with forward Euler. Uses the BELIEVED inertia, never the
//  plant's. State for this model: [phi phi_dot theta theta_dot
//  psi psi_dot], inputs [U2 U3 U4].
// ------------------------------------------------------------

struct LpvMatrices {
	Eigen::Matrix<double, 6, 6> Ad;
	Eigen::Matrix<double, 6, 3> Bd;
	Eigen::Matrix<double, 3, 6> Cd;
};

inline LpvMatrices lpvDiscrete(const InertiaEstimate& I, const State& s, double omega_total, double Ts) {
	const double phi = s(9), theta = s(10);

	// Current Euler rates - the LPV "parameters" the model varies with
	const Eigen::Vector3d rates = eulerRateMatrix(phi, theta) * s.segment<3>(3);
	const double phi_dot = rates(0), theta_dot = rates(1);

	Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
	A(0, 1) = 1.0;
	A(1, 3) = -omega_total * constants::Jtp / I.Ix;
	A(1, 5) = theta_dot * (I.Iy - I.Iz) / I.Ix;
	A(2, 3) = 1.0;
	A(3, 1) = omega_total * constants::Jtp / I.Iy;
	A(3, 5) = phi_dot * (I.Iz - I.Ix) / I.Iy;
	A(4, 5) = 1.0;
	A(5, 1) = (theta_dot / 2.0) * (I.Ix - I.Iy) / I.Iz;
	A(5, 3) = (phi_dot / 2.0) * (I.Ix - I.Iy) / I.Iz;

	Eigen::Matrix<double, 6, 3> B = Eigen::Matrix<double, 6, 3>::Zero();
	B(1, 0) = 1.0 / I.Ix;
	B(3, 1) = 1.0 / I.Iy;
	B(5, 2) = 1.0 / I.Iz;

	LpvMatrices out;
	out.Ad = Eigen::Matrix<double, 6, 6>::Identity() + Ts * A;
	out.Bd = Ts * B;
	out.Cd = Eigen::Matrix<double, 3, 6>::Zero();
	out.Cd(0, 0) = 1.0;	// phi
	out.Cd(1, 2) = 1.0;	// theta
	out.Cd(2, 4) = 1.0;	// psi
	return out;
}


// ------------------------------------------------------------
//  Rotor mixing. U demands -> individual rotor speeds. The mixer
//  matrix is constant so its inverse is computed exactly once.
// ------------------------------------------------------------
struct RotorSpeeds {
	Eigen::Vector4d omega;	// individual rotor speeds [rad/s]
	double omega_total;		// w1 - w2 + w3 - w4 (gyroscopic term)
};

inline RotorSpeeds mixRotors(const Input& u) {
	static const Eigen::Matrix4d M_inv = [] {
		Eigen::Matrix4d M;
		M << 1, 1, 1, 1,
			0, 1, 0, -1,
			-1, 0, 1, 0,
			-1, 1, -1, 1;
		return M.inverse().eval();
		}();

	Eigen::Vector4d UC;
	UC << u(0) / constants::ct,
		u(1) / (constants::ct * constants::arm_length),
		u(2) / (constants::ct * constants::arm_length),
		u(3) / constants::cq;

	const Eigen::Vector4d omega_sq = M_inv * UC;
	if ((omega_sq.array() <= 0.0).any())
		throw std::runtime_error(
			"Negative rotor speed squared - trajectory too aggressive or gains need tuning");

	RotorSpeeds out;
	out.omega = omega_sq.array().sqrt();
	out.omega_total = out.omega(0) - out.omega(1) + out.omega(2) - out.omega(3);
	return out;
}

