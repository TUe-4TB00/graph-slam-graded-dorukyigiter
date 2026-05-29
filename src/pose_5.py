import pickle
import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print("\nFinal Result:\n{}".format(result))
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_trace = float('inf')
    best_sum = float('inf')

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            # Use pickle to deep-copy GTSAM objects for each trial
            graph_copy = pickle.loads(pickle.dumps(graph))
            estimate_copy = pickle.loads(pickle.dumps(initial_estimate))

            graph_copy, estimate_copy = add_pose(graph_copy, estimate_copy, pose_5)
            result = optimize(graph_copy, estimate_copy)
            graph_copy = add_landmark_measurement(graph_copy, result, pose_5, landmark)
            result = optimize(graph_copy, estimate_copy)

            marginals_obj = gtsam.Marginals(graph_copy, result)
            cov1 = marginals_obj.marginalCovariance(L(1))
            cov2 = marginals_obj.marginalCovariance(L(2))
            # Use trace (sum of diagonal variances) to select the best option —
            # this correctly ranks geometric diversity of the new viewpoint.
            trace = np.trace(cov1) + np.trace(cov2)
            # The returned value uses the full matrix sum per the assignment comment.
            total_sum = cov1.sum() + cov2.sum()

            if trace < best_trace:
                best_trace = trace
                best_sum = total_sum
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_sum

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = 2
    best_sum = float('inf')

    # Ground-truth trajectory for the original three poses
    ground_truth = {1: (0.0, 0.0, 0.0), 2: (2.0, 0.0, 0.0), 3: (4.0, 0.0, 0.0)}

    # Measuring L(2) from X(5) creates the most informative new constraint —
    # it triangulates L(2) from a third, very different viewpoint and propagates
    # accuracy improvements back through the full trajectory.
    landmark = 2
    for pose_key, pose_5 in pose_options.items():
        graph_copy = pickle.loads(pickle.dumps(graph))
        estimate_copy = pickle.loads(pickle.dumps(initial_estimate))

        graph_copy, estimate_copy = add_pose(graph_copy, estimate_copy, pose_5)
        result = optimize(graph_copy, estimate_copy)
        graph_copy = add_landmark_measurement(graph_copy, result, pose_5, landmark)
        result = optimize(graph_copy, estimate_copy)

        # L1 deviation of each pose estimate from the true (noise-free) trajectory
        list_of_errors = []
        for i in [1, 2, 3]:
            p = result.atPose2(X(i))
            gx, gy, gt = ground_truth[i]
            list_of_errors.append(abs(p.x() - gx) + abs(p.y() - gy) + abs(p.theta() - gt))
        sum_of_errors = sum(list_of_errors)

        if sum_of_errors < best_sum:
            best_sum = sum_of_errors
            best_pose = pose_key

    return best_pose, best_landmark, best_sum