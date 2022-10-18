# LC2: LiDAR-Camera Loop Constriants From Cross-Modal Place Recognition


To run the procedure, you'll need to generate 2d pose files from best estimates (`predictions.npy`) generated from `test_pr.py`. We provided the dataloader for Vision for Visibility Dataset, assuming they are in processed form.


## Modules

 - `pr_curve.py`: plots pr curve, given numpy files from ``test_pr.py``.

- `filter_and_visualze.py`: gtsam-based pose optimizer using outputs from `test_pr.py` and 2D csv poses generated from rel. pose estimator.


## Required Pose Files
- `loopposes.csv`: Estimated rel.poses from [EPnP](https://github.com/cvlab-epfl/EPnP), given depths from each depth values and local point cloud map. (2D form)
- `odomposes.csv`: Estimated poses from odometry, and projected into 2D.

### Pretrained Dual encoder

``https://drive.google.com/file/d/1c5tRjpAmgoBEXkndTXTj4gixBWOZqYlF/view?usp=sharing``
