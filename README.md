# CSP-pedestrian-pytorch
PyTorch implementation of CSP [https://github.com/liuwei16/CSP]

This code is only for Caltech dataset currently, and only for center-position+height
regression+offset regression model.

We will add Citypersons dataset support in the future.

## Note
On Caltech validation set, we get the best result is 5.84 MR.

## Dependencies

* Python 3.7
* PyTorch 1.5.1 + torchvision 0.6.1
* OpenCV  4.3.0.36
* MMCV 0.6.2

### Installation
1. Get the code.
```
  git clone https://github.com/polariseee/CSP-pedestrian-pytorch.git
```
2. Compile NMS. This code only supports cpu-nms currently.

```
  cd ./external
  python setup.py build_ext --inplace
  rm -rf ./build
```

## Preparation
1. Download the dataset.

 For pedestrian detection, you should firstly download the datasets. For Caltech, we assume the dataset is stored in `./Caltech/`.

2. Dataset preparation.

 For Caltech, the directory structure is
 ```
 *DATA_PATH
    *train
        *IMG
            *set00_V000_I00002.jpg
            *...
        *anno_train10x_alignedby_RotatedFilters
            *set00_V000_I00002.txt
            *...
    *test
        *IMG
            *set06_V000_I00029.jpg
            *...
        * anno_test_1xnew
            *set06_V000_I00029.jpg.txt
            *...
 ```

### Training
1. Train on Caltech
```
  python train.py config/config.py
```
note: This code only supports training with 1 GPU, and we will add multi GPUs support soon.
### Test
1. Caltech
```
  python test.py config/config.py
```

### Evaluation
1. Caltech
 
 You should use matlab to evaluate your results.
 Meantime, Caltech toolbox should be download from official website, and the toolbox is sorted in `./eval_caltech/toolbox`
 Follow the [./eval_caltech/dbEval.m](./eval_caltech/dbEval.m) to get the Miss Rates of detections