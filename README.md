# BiMIn-Depth: Bidirectional Mamba Interaction for Geometric-aware Self-Supervised Monocular Depth Estimation



## Training

To train on KITTI, run:

```bash
python train.py ./args_files/hisfog/kitti/resnet_192x640_train.txt
```
For instructions on downloading the KITTI dataset, see [Monodepth2](https://github.com/nianticlabs/monodepth2)




To train on CityScapes, run:

```bash
python train.py ./args_files/hisfog/cityscapes/args_cityscapes_finetune.txt
```


For preparing cityscapes dataset, please refer to SfMLearner's [prepare_train_data.py](https://github.com/tinghuiz/SfMLearner/blob/master/data/prepare_train_data.py) script.
We used the following command:

```bash
python prepare_train_data.py \
    --img_height 512 \
    --img_width 1024 \
    --dataset_dir <path_to_downloaded_cityscapes_data> \
    --dataset_name cityscapes \
    --dump_root <your_preprocessed_cityscapes_path> \
    --seq_length 3 \
    --num_threads 8
```

## Pretrained weights and evaluation

You can download weights for some pretrained models here:

* [KITTI](https://huggingface.co/MykolaL/SPIdepth/tree/main/kitti)
* [CityScapes](https://huggingface.co/MykolaL/SPIdepth/tree/main/cityscapes)

To evaluate a model on KITTI, run:

```bash
python evaluate_depth_config.py args_files/hisfog/kitti/cvnXt_H_320x1024.txt
```

Make sure you have first run `export_gt_depth.py` to extract ground truth files.

And to evaluate a model on Cityscapes, run:

```bash
python ./tools/evaluate_depth_cityscapes_config.py args_files/args_cvnXt_H_cityscapes_finetune_eval.txt
```

The ground truth depth files can be found at [HERE](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip),
Download this and unzip into `splits/cityscapes`.


**Custom dataset**

You can train on a custom monocular or stereo dataset by writing a new dataloader class which inherits from `MonoDataset` – see the `KITTIDataset` class in `datasets/kitti_dataset.py` for an example.

## Contact us 

If you have any questions, please feel free to contact us: 2222408078@stmail.ujs.edu.cn
~~~
## Acknowledgement
This project is built upon [SQLdepth](https://github.com/hisfog/SfMNeXt-Impl) and [SPIdepth](https://github.com/Lavreniuk/SPIdepth), adopting their settings. and we are grateful for their outstanding contributions.
