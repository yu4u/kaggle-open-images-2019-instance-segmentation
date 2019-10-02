# open-images-2019-instance-segmentation

This repository is a baseline implementation for [Open Images 2019 - Instance Segmentation competition](https://www.kaggle.com/c/open-images-2019-instance-segmentation)
using [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

## Preparation

1. Install maskrcnn_bencmark according to [official guide](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md).
2. Download the [Open Images dataset](https://storage.googleapis.com/openimages/web/download.html) to the project root directory (or make sim link).

```
PROJECT_ROOT
├── README.md
├── config
│   └── e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml
├── create_dataset.py
├── create_submission.py
├── datasets
│   ├── challenge-2019-label300-segmentable-hierarchy.json
│   ├── challenge-2019-train-segmentation-masks.csv
│   ├── test
│   ├── train
│   └── train_masks
├── test.py
├── train.py
└── util.py
```

- `test`: test images (99,999 images)
- `train`: train images
- `train_masks`: train mask images

Trained models are available from [Kaggle Dataset](https://www.kaggle.com/ren4yu/openimages2019instancesegmentationmodels).
If you use the trained models, please skip to the 'Test for Layer 0 Classes' section.

## Train on Open Images Dataset

### Create Dataset for Layer 0 Classes

Create COCO format dataset for layer 0 class.

```bash
python create_dataset.py -l 0
```

The COCO format dataset is created as:

```
PROJECT_ROOT
├── datasets
│   └── coco
│       ├── annotations
│       └── train2017
```

This is the COCO-based format, thus it can be used on the other library like [mmdetection](https://github.com/open-mmlab/mmdetection) (but not tested).

### Train for Layer 0 Classes

Train on 8GPUs. This requires only 14 hours using V100 8GPUs

```bash
python -m torch.distributed.launch --nproc_per_node=8 train.py OUTPUT_DIR "layer0"
```

Train on a single GPU. This requires about 4 days using a V100 GPU.

```bash
python train.py --config-file config/e2e_mask_rcnn_X_101_32x8d_FPN_1x_1gpu.yaml OUTPUT_DIR "layer0"
```

Training steps can be reduced without large degradation of accuracy.
The following should requires only a day for training with a single GPU.

```bash
python train.py --config-file config/e2e_mask_rcnn_X_101_32x8d_FPN_1x_1gpu.yaml OUTPUT_DIR "layer0" SOLVER.STEPS "(70000, 100000)" SOLVER.MAX_ITER 120000
```

### Test for Layer 0 Classes

```bash
python test.py -l 0 --weight [TRAINED_WEIGHT_PATH (e.g. layer0/model_0060000.pth)]
```

The resulting files will created as:

```
PROJECT_ROOT
├── datasets
│   └── test_0_results
```

### Create Submission File for Layer 0 Classes

```bash
python create_submission.py -l 0
```

### Create Submission File for Layer 1 Classes

Do the same procedure also for layer 1 classes:

```bash
python create_dataset.py -l 1  # this overwrite layer 0 dataset. Please move it if needed later
python -m torch.distributed.launch --nproc_per_node=8 train.py OUTPUT_DIR "layer1"
python test.py -l 1 --weight [TRAINED_WEIGHT_PATH (e.g. layer1/model_0060000.pth)]
python create_submission.py -l 1
```

### Integrate Two Submission Files

```bash
python integrate_results.py --input1 output_0.csv --input2 output_1.csv
```

OK, let's submit the resulting file `integrated_result.csv` !

