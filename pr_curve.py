import gtsam
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from vistools import *
from sklearn.metrics import precision_recall_curve, classification_report

# 0. load files
root_folder = 'results'

# sequence_c = 'results_lc2_night'
sequence_c = 'results_lc2_day'
distance_c = np.load(os.path.join(root_folder, sequence_c, "distance.npy"))
utmQ_c = np.load(os.path.join(root_folder, sequence_c, "qlocation.npy"))
utmDb_c = np.load(os.path.join(root_folder, sequence_c, "Dblocation.npy"))
predictions_c = np.load(os.path.join(root_folder, sequence_c, "predictions.npy"))
iscorrect_c = np.load(os.path.join(root_folder, sequence_c, "iscorrect.npy"))

# sequence_i = 'results_img_night'
sequence_i = 'results_img_day'
distance_i = np.load(os.path.join(root_folder, sequence_i, "distance.npy"))
utmQ_i = np.load(os.path.join(root_folder, sequence_i, "qlocation.npy"))
utmDb_i = np.load(os.path.join(root_folder, sequence_i, "Dblocation.npy"))
predictions_i = np.load(os.path.join(root_folder, sequence_i, "predictions.npy"))
iscorrect_i = np.load(os.path.join(root_folder, sequence_i, "iscorrect.npy"))

topN = 1
precision_c, recall_c, _ = precision_recall_curve(iscorrect_c[:, 0:topN].reshape(-1), 1/distance_c[:, 0:topN].reshape(-1))
precision_i, recall_i, _ = precision_recall_curve(iscorrect_i[:, 0:topN].reshape(-1), 1/distance_i[:, 0:topN].reshape(-1))


fig, ax = plt.subplots()
ax.plot(recall_i, precision_i, color=np.array([255/255, 194/255, 60/255]), label='VPR-based')
ax.plot(recall_c, precision_c, color=np.array([251/255, 37/255, 118/255]), label='cross-matching-based')

ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
ax.set_aspect('equal')
plt.legend()
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
