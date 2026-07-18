
#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <cmath>
#include "Trajectories.hpp"
#include "Simulation.hpp"

// ============================================================
//  Csv - result output and error metrics.
//
//  No plotting in C++ - results go to CSV and the Python plotter
//  reads them.
// ============================================================

// Linear interpolation of an outer-grid signal onto the inner time axis
inline Eigen::ArrayXd interpToInner(const Eigen::ArrayXd& t_outer,
	const Eigen::ArrayXd& values,
	const Eigen::ArrayXd& t_inner) {
	Eigen::ArrayXd out(t_inner.size());
	Eigen::Index j = 0;
	for (Eigen::Index i = 0; i < t_inner.size(); ++i) {
		const double t = t_inner(i);
		while (j + 1 < t_outer.size() - 1 && t_outer(j + 1) < t) ++j;
		const double t0 = t_outer(j), t1 = t_outer(j + 1);
		const double frac = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0;
		out(i) = values(j) + frac * (values(j + 1) - values(j));
	}
	return out;
}

// 3D RMS tracking error against the (interpolated) reference
inline double rmsTrackingError(const SimResult& res,
	const TrajectoryData& ref,
	const Eigen::ArrayXd& t_outer) {
	const Eigen::ArrayXd xr = interpToInner(t_outer, ref.x, res.time);
	const Eigen::ArrayXd yr = interpToInner(t_outer, ref.y, res.time);
	const Eigen::ArrayXd zr = interpToInner(t_outer, ref.z, res.time);

	const Eigen::ArrayXd ex = xr - res.states.col(6).array();
	const Eigen::ArrayXd ey = yr - res.states.col(7).array();
	const Eigen::ArrayXd ez = zr - res.states.col(8).array();

	return std::sqrt((ex.square() + ey.square() + ez.square()).mean());
}

inline void writeCsv(const std::string& path, const SimResult& res,
	const TrajectoryData& ref, const Eigen::ArrayXd& psi_ref,
	const Eigen::ArrayXd& t_outer) {
	std::ofstream f(path);
	f << "t,u,v,w,p,q,r,x,y,z,phi,theta,psi,"
		"U1,U2,U3,U4,Ix_hat,Iy_hat,Iz_hat,"
		"x_ref,y_ref,z_ref,psi_ref\n";

	const Eigen::ArrayXd xr = interpToInner(t_outer, ref.x, res.time);
	const Eigen::ArrayXd yr = interpToInner(t_outer, ref.y, res.time);
	const Eigen::ArrayXd zr = interpToInner(t_outer, ref.z, res.time);
	const Eigen::ArrayXd pr = interpToInner(t_outer, psi_ref, res.time);

	for (Eigen::Index i = 0; i < res.time.size(); ++i) {
		f << res.time(i);
		for (int c = 0; c < 12; ++c) f << ',' << res.states(i, c);
		for (int c = 0; c < 4; ++c)  f << ',' << res.inputs(i, c);
		for (int c = 0; c < 3; ++c)  f << ',' << res.believed(i, c);
		f << ',' << xr(i) << ',' << yr(i) << ',' << zr(i) << ',' << pr(i) << '\n';
	}
}