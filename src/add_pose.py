
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # Robot starts at X(3) = (4, 0, 0), rotates 45° CCW, moves 2m, rotates 45° CCW more.
    # Final global position: X(4) = (4+√2, √2, π/2).
    # Odometry in X(3)'s local frame (theta=0): dx=√2, dy=√2, dtheta=π/2.
    odometry = gtsam.Pose2(math.sqrt(2), math.sqrt(2), math.pi / 2)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))
    initial_estimate.insert(X(4), gtsam.Pose2(4.0 + math.sqrt(2), math.sqrt(2), math.pi / 2))
    return graph, initial_estimate