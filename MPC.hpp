
#pragma once
#include <Eigen/Dense>
#include "Config.hpp"
#include "DroneModel.hpp"

// ============================================================
//  Mpc - the inner loop attitude controller.
//
//  Standard condensed MPC on the augmented LPV system. The state
//  is augmented with the previous inputs so the optimiser works
//  in input INCREMENTS (du), which gives integral action for free.
//
//  Augmented state (9): [phi phi_dot theta theta_dot psi psi_dot U2 U3 U4]
//  Decision variables:  du over the horizon (3 * hz)
//
//  The horizon shrinks near the end of the reference so hz is a
//  runtime argument, and the QP matrices are rebuilt per call.
//  ldlt solve instead of an explicit inverse - same maths, better
//  conditioning.
// ============================================================

class MpcController {
public:
	explicit MpcController(const ControlParams& ctrl) : ctrl_{ ctrl } {}

	// Returns only the first input increment (receding horizon)
	Eigen::Vector3d solve(const LpvMatrices& lpv,
		const Eigen::Matrix<double, 9, 1>& x_aug,
		const Eigen::VectorXd& ref_window,
		int hz) const {
		constexpr int nx = 6, nu = 3, ny = 3;
		constexpr int na = nx + nu;	// 9

		// --- Augmented system: track du instead of u -------------------
		Eigen::Matrix<double, na, na> A_aug = Eigen::Matrix<double, na, na>::Zero();
		A_aug.topLeftCorner<nx, nx>() = lpv.Ad;
		A_aug.topRightCorner<nx, nu>() = lpv.Bd;
		A_aug.bottomRightCorner<nu, nu>().setIdentity();

		Eigen::Matrix<double, na, nu> B_aug;
		B_aug.topRows<nx>() = lpv.Bd;
		B_aug.bottomRows<nu>().setIdentity();

		Eigen::Matrix<double, ny, na> C_aug = Eigen::Matrix<double, ny, na>::Zero();
		C_aug.leftCols<nx>() = lpv.Cd;

		// --- Weight blocks (diagonals expanded here, nowhere else) -----
		const Eigen::Matrix3d Q = ctrl_.Q.asDiagonal();
		const Eigen::Matrix3d S = ctrl_.S.asDiagonal();
		const Eigen::Matrix3d R = ctrl_.R.asDiagonal();

		const Eigen::Matrix<double, na, na> CQC = C_aug.transpose() * Q * C_aug;
		const Eigen::Matrix<double, na, na> CSC = C_aug.transpose() * S * C_aug;
		const Eigen::Matrix<double, ny, na> QC = Q * C_aug;
		const Eigen::Matrix<double, ny, na> SC = S * C_aug;

		// --- Stacked horizon matrices ----------------------------------
		Eigen::MatrixXd Qdb = Eigen::MatrixXd::Zero(na * hz, na * hz);
		Eigen::MatrixXd Tdb = Eigen::MatrixXd::Zero(ny * hz, na * hz);
		Eigen::MatrixXd Rdb = Eigen::MatrixXd::Zero(nu * hz, nu * hz);
		Eigen::MatrixXd Cdb = Eigen::MatrixXd::Zero(na * hz, nu * hz);
		Eigen::MatrixXd Adc = Eigen::MatrixXd::Zero(na * hz, na);

		Eigen::Matrix<double, na, na> A_pow = A_aug;	// A^(i+1) running product
		for (int i = 0; i < hz; ++i) {
			const bool last = (i == hz - 1);
			Qdb.block<na, na>(na * i, na * i) = last ? CSC : CQC;
			Tdb.block<ny, na>(ny * i, na * i) = last ? SC : QC;
			Rdb.block<nu, nu>(nu * i, nu * i) = R;

			// Cdb row i: [A^(i-j) B] for j <= i, built without matrix_power
			Eigen::Matrix<double, na, nu> AjB = B_aug;
			for (int j = i; j >= 0; --j) {
				Cdb.block<na, nu>(na * i, nu * j) = AjB;
				if (j > 0) AjB = A_aug * AjB;
			}

			Adc.block<na, na>(na * i, 0) = A_pow;
			if (!last) A_pow = A_aug * A_pow;
		}

		// --- Condensed QP ----------------------------------------------
		const Eigen::MatrixXd Hdb = Cdb.transpose() * Qdb * Cdb + Rdb;

		// ft = [x_aug; ref]^T * [Adc^T Qdb Cdb ; -Tdb Cdb]
		const Eigen::VectorXd ft =
			(Adc.transpose() * Qdb * Cdb).transpose() * x_aug
			- (Tdb * Cdb).transpose() * ref_window;

		const Eigen::VectorXd du = -Hdb.ldlt().solve(ft);
		return du.head<3>();
	}

private:
	ControlParams ctrl_;
};