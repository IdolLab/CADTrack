# [AAAI2026]CADTrack: Learning Contextual Aggregation with Deformable Alignment for Robust RGBT Tracking

![pipline](assets/pipline.png)

> [CADTrack: Learning Contextual Aggregation with Deformable Alignment for Robust RGBT Tracking](https://arxiv.org/abs/2511.17967)  
> Hao Li, Yuhao Wang, Xiantao Hu, Wenning Hao, Pingping Zhang, Dong Wang, Huchuan Lu
> AAAI 2026

Official implementation of [**CADTrack**](https://arxiv.org/pdf/2511.17967), including [Models & Raw Results](https://pan.baidu.com/s/1TR4qnWtXS140pddngcn_-w 
)(Baidu Driver:9527) and training&testing codes.
 
## Installation
Create and activate a conda environment:
```
conda create -n CADTrack python=3.10
conda activate CADTrack
```
Install the required packages:
```
bash install_cadtrack.sh
```

## Data Preparation
[GTOT & RGBT210 & RGBT234 & LasHeR](https://chenglongli.cn/Datasets-and-benchmark-code/),[VTUAV](https://zhang-pengyu.github.io/DUT-VTUAV/) Put the datasets in ./data/. It should look like:
```
$<PATH_of_STTrack>
-- data
    -- GTOT
        |-- BlackCar
        |-- Black5wan1
        ...
    -- RGBT210
        |-- afterrain
        |-- aftertree
        ...
    -- RGBT234
        |-- afterrain
        |-- aftertree
        ...
    -- LasHeR/train
        |-- 1boygo
        |-- 1handsth
        ...
    -- LasHeR/test
        |-- 1blackteacher
        |-- 1boycoming
        ...
    -- VTUAV/train
        |-- animal_002
        |-- bike_002
        ...
    -- VTUAV/test_ST
        |-- animal_001
        |-- bike_003
        ...
    -- VTUAV/test_LT
        |-- animal_003
        |-- animal_004
        ...
```

## Path Setting
Run the following command to set paths:
```
cd <PATH_of_CADTrack>
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

## Training
Dowmload the pretrained [foundation model](https://pan.baidu.com/s/15GjTLQboXcfJaTD5sLLRDQ?pwd=hmaa) (Baidu Driver:hmaa) 
and put it under ./pretrained/.
```
bash train.sh
```
You can train models with various modalities and variants by modifying ```train.sh```.

## Testing
[GTOT & RGBT210 & RGBT234 & LasHeR & VTUAV] \
Modify the <DATASET_PATH> and <SAVE_PATH> in```./RGBT_workspace/test_rgbt_mgpus.py```, then run:
```
bash test.sh
```
We refer you to [Evaluation Toolkit](https://chenglongli.cn/Datasets-and-benchmark-code/) for GTOT RGBT210 RGBT234 LasHeR evaluation, 
and refer you to [VTUAV_Evaluation](https://zhang-pengyu.github.io/DUT-VTUAV/) for VTUAV evaluation.

## **Poster** 📜
<p align="center">
    <img src="assets/Poster.png" alt="Poster" style="width:100%;">
</p>

## Bixtex
If you find CADTrack is helpful for your research, please consider citing:

```bibtex
@inproceedings{cadtrack,
  title={CADTrack: Learning Contextual Aggregation with Deformable Alignment for Robust RGBT Tracking},
  author={Hao Li and Yuhao Wang and Xiantao Hu and Wenning Hao and Pingping Zhang and Dong Wang and Huchuan Lu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  year={2026}
}
```

## Acknowledgment
- This repo is based on [STTrack](https://github.com/NJU-PCALab/STTrack) and [IDEA](https://github.com/924973292/IDEA) which are excellent works.
- We thank for the [PyTracking](https://github.com/visionml/pytracking) library, which helps us to quickly implement our ideas.
