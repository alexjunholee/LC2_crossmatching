import gtsam
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from vistools import *

# 0. load files
root_folder = 'results'
sequence = 'lc2_day'
# sequence = 'results_img_day'

distance = np.load(os.path.join(root_folder, sequence, "distance.npy"))
utmQ = np.load(os.path.join(root_folder, sequence, "qlocation.npy"))
utmDb = np.load(os.path.join(root_folder, sequence, "Dblocation.npy"))
predictions = np.load(os.path.join(root_folder, sequence, "predictions.npy"))
iscorrect = np.load(os.path.join(root_folder, sequence, "iscorrect.npy"))
odomposes = os.path.join(root_folder, sequence, "odomposes.csv") # arragnged 2d poses, [x y t]
loopposes = os.path.join(root_folder, sequence, "loopposes.csv") # arragnged 2d rel. poses from top 1, [x y t]
if os.path.isfile(odomposes):
    odomposes = np.loadtxt(odomposes, delimiter=",")
else:
    odomposes = None
if os.path.isfile(loopposes):
    loopposes = np.loadtxt(loopposes, delimiter=",")
else:
    loopposes = None

numQ = utmQ.shape[0]
numDb = utmDb.shape[0]

###### 1. visualize before optimization
fig = plt.figure(tight_layout={'pad': 0})
ax = fig.gca(projection='3d')
ax.scatter(utmDb[:, 0], utmDb[:, 1], np.zeros(utmDb[:, 0].shape), s=1, color=np.array([0, 0, 0]))
ax.scatter(utmQ[:, 0], utmQ[:, 1], 1500*np.ones(utmQ[:, 0].shape), s=1, color=np.array([0, 0, 0]))
    
set_axes_equal(ax)
ax.view_init(elev=0, azim=270)
# ax.view_init(elev=40, azim=250)
for q in range(numQ):
    nthbest = 0
    while nthbest < 10 and distance[q, nthbest] < 0.1:
     db_idx = predictions[q, nthbest]
     p1 = [utmQ[q, 0], utmQ[q, 1], 1500]
     p2 = [utmDb[db_idx, 0], utmDb[db_idx, 1], 0]
     x12 = np.linspace(p1[0], p2[0], 1000)
     y12 = np.linspace(p1[1], p2[1], 1000)
     z12 = np.linspace(p1[2], p2[2], 1000)
     if iscorrect[i, nthbest]:
         ax.plot(x12, y12, z12, linewidth=0.8, color=np.array([0, 1, 0]))
     else:
         ax.plot(x12, y12, z12, linewidth=0.5, color=np.array([1, 0, 0]))
    nthbest += 1
plt.axis('off')
plt.show()

###### 2. optimize through gtsam
poseDb = np.concatenate((utmDb, np.zeros((utmDb.shape[0], 1))), axis=1)
if odomconst is None:  # if no odometry, generate odometry with initial heading error
    initial_heading_error = 0.54
    utmQ_r = np.transpose(np.dot([[np.cos(initial_heading_error), -np.sin(initial_heading_error)],
                                  [np.sin(initial_heading_error), np.cos(initial_heading_error)]], np.transpose(utmQ)))
    poseQ = np.concatenate((utmQ_r, initial_heading_error * np.ones((utmQ_r.shape[0], 1))), axis=1)
else:
    poseQ = loopposes

knownModel = gtsam.noiseModel.Diagonal.Variances(vector3(1e-6, 1e-6, 1e-6))
odomModel = gtsam.noiseModel.Diagonal.Variances(vector3(1e-2, 1e-2, 1e-2))
loopModel = gtsam.noiseModel.Diagonal.Variances(vector3(1e+4, 1e+4, 5e+4))
robustLoopModel = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy.Create(1), loopModel)
robustOdomModel = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy.Create(1), odomModel)

graph = gtsam.NonlinearFactorGraph()
prev_pose = gtsam.Pose2(0, 0, initial_heading_error)
initial = gtsam.Values()
initial.insert(0, prev_pose)
np.random.seed()
for factor_idx in range(numQ-1):
    Pose0 = gtsam.Pose2(poseQ[factor_idx, 0], poseQ[factor_idx, 1], poseQ[factor_idx, 2])
    Pose1 = gtsam.Pose2(poseQ[factor_idx+1, 0], poseQ[factor_idx+1, 1], poseQ[factor_idx, 2])
    delta = Pose0.inverse() * Pose1
    # generate odom error
    # np.random.seed(factor_idx) # to produce "consistent random odom error"
    noise_x = 5e-2 * np.random.rand(1)
    noise_y = 5e-2 * np.random.rand(1)
    noise_t = 2e-3 * np.random.rand(1)
    delta = gtsam.Pose2(delta.x()+noise_x, delta.y()+noise_y, noise_t)
    # add odom factors
    graph.add(gtsam.BetweenFactorPose2(factor_idx, factor_idx+1, delta, robustOdomModel))
    # add erroneous odom poses
    prev_pose = delta * prev_pose
    initial.insert(factor_idx+1, prev_pose)

random_percentage = 0
# add loop factors
for factor_idx in range(numQ):
    loop_idx = predictions[factor_idx, 0]
    if loopposes is None: #just assign zero-pose if there's no rel info
        delta = gtsam.Pose2(0, 0, 0)
    else:
        delta = gtsam.Pose2(loopposes[loop_idx, 0], loopposes[loop_idx, 1], loopposes[loop_idx, 2])
    # add false(random) loop factors from this node
    if np.random.rand(1) < random_percentage:
        random_idx = int(numDb * np.random.rand(1))
        graph.add(gtsam.BetweenFactorPose2(factor_idx, numQ + random_idx, delta, robustLoopModel))
        iscorrect[factor_idx, 0] = False
    else:
        graph.add(gtsam.BetweenFactorPose2(factor_idx, numQ + loop_idx, delta, robustLoopModel))

# prior factor
for factor_idx in range(numDb):
    Posegt = gtsam.Pose2(poseDb[factor_idx, 0], poseDb[factor_idx, 1], poseDb[factor_idx, 2])
    graph.add(gtsam.PriorFactorPose2(numQ + factor_idx, Posegt, knownModel))

# initialize GT nodes
for pose_db in range(numDb):
    pose = gtsam.Pose2(poseDb[pose_db, 0], poseDb[pose_db, 1], poseDb[pose_db, 2])
    initial.insert(pose_db + numQ, pose)

# optimizer
# graph.print("Factor graph:\n")
print("optimizing...")
params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
result = optimizer.optimize()
out_graph = optimizer.iterate()
# print("initial error = ", graph.error(initial))
# print("final error = ", graph.error(result))
info = np.zeros((numQ, 1))
for q_idx in range(numQ-1, 2*numQ-1):
    info[q_idx-numQ+1] = np.linalg.norm(out_graph.at(q_idx).information()[0:2, 0:2]) # information on position

###### 3. visualize after optimization
point = np.zeros((numQ, 2))
for factor_idx in range(numQ):
    point[factor_idx] = [optimizer.values().atPose2(factor_idx).x(), optimizer.values().atPose2(factor_idx).y()]

city = [520, 100]
campus = [710, -245]
offset = city
noisy = np.zeros((numQ, 2))
for factor_idx in range(numQ):
    noisy[factor_idx] = [initial.atPose2(factor_idx).x() + offset[0], initial.atPose2(factor_idx).y() + offset[1]]  # just for good look

fig = plt.figure(tight_layout={'pad': 0})
ax = fig.gca(projection='3d')

# plot_z_gap = 0
# ax.view_init(elev=90, azim=-79)
plot_z_gap = 500
ax.view_init(elev=33, azim=-79)
print("Top-1 Precision: %.3f" % np.mean(iscorrect[:, 0]))
ax.scatter(noisy[:, 0], noisy[:, 1], plot_z_gap*np.ones(noisy[:, 0].shape), s=1, color=np.array([63/255, 0, 113/255]), label='Odom_noise')
# ax.scatter(point[:, 0], point[:, 1], plot_z_gap*np.ones(point[:, 0].shape), s=1, color=np.array([255/255, 194/255, 60/255]), label='VPR-based')
ax.scatter(point[:, 0], point[:, 1], plot_z_gap*np.ones(point[:, 0].shape), s=1, color=np.array([251/255, 37/255, 118/255]), label='cross-matching-based')
ax.scatter(utmDb[:, 0], utmDb[:, 1], np.zeros(utmDb[:, 0].shape), s=1, color=np.array([0, 0, 0]), label='Database')
plt.legend()
n_loops_aft_optimization = 0
n_truth_aft_optimization = 0
# th = 0.698
th = 0.078
for i in range(numQ):
    j = 0
    while j < 1 and distance[i, j] < th:
        db_i = predictions[i, j]
        p1 = [point[i, 0], point[i, 1], plot_z_gap]
        p2 = [utmDb[db_i, 0], utmDb[db_i, 1], 0]
        x12 = np.linspace(p1[0], p2[0], plot_z_gap)
        y12 = np.linspace(p1[1], p2[1], plot_z_gap)
        z12 = np.linspace(p1[2], p2[2], plot_z_gap)
        if info[i] > 0.00012:
            n_loops_aft_optimization += 1
            if iscorrect[i, j]:
                n_truth_aft_optimization += 1
                ax.plot(x12, y12, z12, linewidth=0.8, color=np.array([0, 1, 0]))
            else:
                ax.plot(x12, y12, z12, linewidth=0.5, color=np.array([1, 0, 0]))
        j += 1

print("Valid LC.: %d out of %d" % (n_truth_aft_optimization, n_loops_aft_optimization))
print("RMSE: %.2f" % (np.sqrt(np.mean(np.linalg.norm(utmQ - point, axis=1) * np.linalg.norm(utmQ - point, axis=1)))))
set_axes_equal(ax)
plt.axis('off')
plt.show()
