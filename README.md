# Fast-BEV
[Fast-BEV: A Fast and Strong Bird’s-Eye View Perception Baseline](https://arxiv.org/abs/2301.12511)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/fast-bev++.png)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/benchmark_setting.png)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/benchmark.png)

## Usage

<<<<<<< HEAD
### Installation

* CUDA>=9.2, GCC>=5.4, Python >= 3.6, Pytorch >= 1.8.1, Torchvision >= 0.9.1

* MMCV-full == 1.4.0, MMDetection == 2.14.0, MMSegmentation == 0.14.1

    ```bash
    # gcc >= 5.4 

    cd env/mmcv
    # TODO
    MMCV_OPS=1 pip install -v . --user

    cd ../mmdetection
    pip install -v -e . --user

    cd ../mmsegmentation
    pip install -v -e . --user

    cd ../../
    pip install -v -e . --user 
    ```

* Other requirements

    ```bash
    pip install -r requirements.txt --user
    ```

### Dataset preparation

Please download nuscenes dataset and organize them as follows:

```
TODO
```

If you are using ceph, you can change the arguments in the configuration. 

e.g.

```python
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        data_root: 'ceph:s3://path/to/data'}))

train_pipeline = [
    dict(
        type='MultiViewPipeline', 
        sequential=True,
        n_images=6,
        n_times=4,
        transforms=[
            dict(
                type='LoadImageFromFile',
                file_client_args=file_client_args),
        ]),
    ...
]
```

### Training

We provide several configs in `configs/fastbev/exp/paper`.

Configure the `tools/fastbev_run.sh` script like

```bash
slurm_train $PARTITION 32 paper/<CONFIG_NAME>
```

And run 

```
sh tools/fastbev_run.sh <PARTITION>
```

### Evaluation

* Inference

    Configure the `tools/fastbev_run.sh` script like

    ```bash
    slurm_test $PARTITION 16 paper/<CONFIG_NAME>
    ```

    ```
    sh tools/fastbev_run.sh <PARTITION>
    ```

* Evaluation

    Configure the `tools/fastbev_run.sh` script like

    ```bash
    slurm_eval $PARTITION 1 paper/<CONFIG_NAME>
    ```

    ```
    sh tools/fastbev_run.sh <PARTITION>
    ```

### Deployment

TODO
=======
<!-- ## latency on device -->
>>>>>>> df95a4236b32d835c1df0070a0e72ac2bd7962f1

## Citation
```
@article{li2023fast,
  title={Fast-BEV: A Fast and Strong Bird's-Eye View Perception Baseline},
  author={Li, Yangguang and Huang, Bin and Chen, Zeren and Cui, Yufeng and Liang, Feng and Shen, Mingzhu and Liu, Fenggang and Xie, Enze and Sheng, Lu and Ouyang, Wanli and others},
  journal={arXiv preprint arXiv:2301.12511},
  year={2023}
}
```

