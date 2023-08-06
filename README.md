# LC2: LiDAR-Camera Loop Constriants From Cross-Modal Place Recognition (RA-L 2023)


## How to run

1. Preprocess the ViViD to contain ```imglist.txt``` and ```gpslist.txt```

2. run ```manydepth``` by following command:

    ```
    python -m manydepth.test_all --target_image_path /your/vivid/path --intrinsics_json_path vivid_sequence_intrinsics.json --model_path manydepth/pretrained
    ```

    The script above will generate ```.npy``` and ```.png```, representing depth in absolute scale and relative scale.

3. run ```lc2``` by following command:

    ```
    python test_pr.py
    ```

#### Additional Funtionalities

 - `pr_curve.py`: plots pr curve, given numpy files from ``test_pr.py``.

- `filter_and_visualze.py`: gtsam-based pose optimizer using outputs from `test_pr.py` and 2D csv poses generated from rel. pose estimator.

** to run `filter_and_visualize`, you need the pose files below:
  `loopposes.csv`: Estimated rel.poses from [EPnP](https://github.com/cvlab-epfl/EPnP), given depths from each depth values and local point cloud map. (2D form)
  `odomposes.csv`: Estimated poses from odometry, and projected into 2D.

### Pretrained weights

1. Dual encoder [[google drive](https://drive.google.com/file/d/1c5tRjpAmgoBEXkndTXTj4gixBWOZqYlF/view?usp=sharing)]

2. Manydepth trained on ViViD [[google drive](https://drive.google.com/file/d/1DCdmDYQMojCUDYqzooHiPTIL9NwDBegK/view?usp=sharing)]



###Codebases

[1] Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition [(github)](https://github.com/QVPR/Patch-NetVLAD)

[2] The Temporal Opportunist: Self-Supervised Multi-Frame Monocular Depth [(github)](https://github.com/nianticlabs/manydepth)



###Bibtex
```
@ARTICLE{lee2023lc2,
  author={Lee, Alex Junho and Song, Seungwon and Lim, Hyungtae and Lee, Woojoo and Myung, Hyun},
  journal={IEEE Robotics and Automation Letters}, 
  title={(LC)$^{2}$: LiDAR-Camera Loop Constraints for Cross-Modal Place Recognition}, 
  year={2023},
  volume={8},
  number={6},
  pages={3589-3596},
  doi={10.1109/LRA.2023.3268848}}
```