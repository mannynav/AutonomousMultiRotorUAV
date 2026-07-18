
#include <iostream>
#include <iomanip>
#include <future>
#include "Config.hpp"
#include "Trajectories.hpp"
#include "Simulation.hpp"
#include "Csv.hpp"

// ============================================================
//  main - three runs:
//
//  1) Nominal      - controller and plant agree. Baseline sanity.
//  2) Perturbed    - plant inertia differs from what the controller
//                    believes, no adaptation.
//  3) Perturbed+RLS - same wrong start, RLS corrects the belief
//                    online.
//
//  Runs 2 and 3 are independent so they execute concurrently via
//  std::async - each gets its own Simulation instance, no shared
//  mutable state between them.
// ============================================================

int main() {

	Config cfg;
	cfg.sim.trajectory_id = 4;	// 1=Helix 2=StraightLine 3=Wave 4=VerticalWave

	// Build the reference once, shared read-only by all runs
	const Eigen::ArrayXd t = makeTimeVector(cfg);
	const double delta_height = cfg.trajectory.final_height - cfg.trajectory.initial_height;
	const AnyTrajectory traj = makeTrajectory(cfg.sim.trajectory_id);
	const TrajectoryData ref = generate(traj, cfg.trajectory, t, delta_height);
	const Eigen::ArrayXd psi_ref = computeYawReference(ref);

	std::cout << std::fixed << std::setprecision(4);
	std::cout << "Trajectory " << cfg.sim.trajectory_id
		<< ", " << t.size() << " outer steps, "
		<< cfg.sim.total_time << " s\n\n";

	// ---- Run 1: nominal ---------------------------------------------------
	{
		Simulation<RigidBodyPlant> sim{ cfg, RigidBodyPlant{ cfg.physical } };
		const SimResult res = sim.run(ref, psi_ref, t);
		std::cout << "Nominal          RMS = "
			<< rmsTrackingError(res, ref, t) << " m\n";
		writeCsv("nominal.csv", res, ref, psi_ref, t);
	}

	// ---- Runs 2 + 3: perturbed plant, fixed vs RLS, concurrently ----------
	// True inertia the controller doesn't know about: 30% heavier in
	// roll/pitch, 20% lighter in yaw.

	PhysicalParams true_params = cfg.physical;
	true_params.Ix *= 1.3;
	true_params.Iy *= 1.3;
	true_params.Iz *= 0.8;

	auto fut_fixed = std::async(std::launch::async, [&] {
		Simulation<RigidBodyPlant> sim{ cfg, RigidBodyPlant{ true_params } };
		return sim.run(ref, psi_ref, t);
		});

	auto fut_rls = std::async(std::launch::async, [&] {
		Simulation<RigidBodyPlant, RlsEstimator> sim{
			cfg, RigidBodyPlant{ true_params }, RlsEstimator{ cfg.control.Ts } };
		return sim.run(ref, psi_ref, t);
		});

	const SimResult res_fixed = fut_fixed.get();
	const SimResult res_rls = fut_rls.get();

	const double rms_fixed = rmsTrackingError(res_fixed, ref, t);
	const double rms_rls = rmsTrackingError(res_rls, ref, t);

	std::cout << "Perturbed fixed  RMS = " << rms_fixed << " m\n";
	std::cout << "Perturbed RLS    RMS = " << rms_rls << " m  ("
		<< std::showpos << (rms_fixed - rms_rls) / rms_fixed * 100.0
		<< std::noshowpos << "% improvement)\n\n";

	// Where did the belief end up vs the truth
	const auto last = res_rls.believed.rows() - 1;
	std::cout << "Believed inertia at end of flight (true in brackets):\n"
		<< "  Ix = " << res_rls.believed(last, 0) << "  (" << true_params.Ix << ")\n"
		<< "  Iy = " << res_rls.believed(last, 1) << "  (" << true_params.Iy << ")\n"
		<< "  Iz = " << res_rls.believed(last, 2) << "  (" << true_params.Iz << ")\n";

	writeCsv("perturbed_fixed.csv", res_fixed, ref, psi_ref, t);
	writeCsv("perturbed_rls.csv", res_rls, ref, psi_ref, t);

	return 0;
}