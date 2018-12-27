# Requirement
Main requirement:
* Pytorch 0.4.1

If you miss any package, please install it by: `pip install missing_package`

# Data split
Change following parameters:  
* `skiprows`: Number of row you want to skip  
Ex: If you want to skip 30k data, `skiprows=(1,30000)`
* `nrows`: Number of data / class  
Default: 50000 data / class
* `root_csv`: Directory of your `train_simplified` folder  
Ex: 
`/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/csv/train_simplified/`
* `split_csv`: Directory where you want to save splited data into  
Ex:  
`/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/50k/`


Run:  
`python split_data_top.py`  
Output:  
There are 340 csv files of `train` and `valid` are saved at your `split_csv`. Each csv file has `nrows` data.

# Run model  
* Configure `train.yml`  
In this file, please change the `main parameters` as following:  
  * `train_split`  
  Path to `train` folder: `{split_csv}/train` 
  
  * `train_token`  
  Dont care, but it is same as `train_split`
  * `valid_split`  
  Path to `valid` folder: `{split_csv}/valid`
  * `valid_token`  
  Dont care, but it is same as `valid_split`  

  You can change other parameters `workers`, `batch_size`, ... to be suiatable for your environement
  
* Run  
     ```bash
     bash run_model.sh
     ```
  Log and checkpoints will be saved to `./logs/se_resnext101_50k`. Change it as you want
  
# Predict model  
* Configure `inference.yml`  
In this file, please change:  
  * `infer_csv`  
  Path to your `test_simplified.csv` file
  
* Run
     ```bash
     bash predict_5best.sh
     ```
  We use multiple checkpoints (snapshot) during training. Ensembling 5 best checkpoints will give
  free 0.0005 boost.  
  Outputs are the `logits` will be saved into your `log` folder that you defined above
 
# Predict dataset for cleaning 
* Configure `inference.yml`  
In this file, please change:  
  * `infer_csv`  
  Comment this line
  * `data_clean_train`  
  Path to train data you want to clean
  * `data_clean_valid`  
  Path to valid data you want to clean
  
* Run
     ```bash
     bash predict_data_for_clean.sh
     ```
  Please change to the best checkpoint of your model you use for clean data  
  Ex: `LOGDIR=$(pwd)/logs/clean_model_2_resnet34/`

# Clean data
In this file, change following parameter correct to your environment  
* `data_clean_train`  
Path to `train data` you want to clean.  
Ex: `/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/30k/data_2/train/`

* `data_clean_valid`  
Path to `valid data` you want to clean.  
Ex: `/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/30k/data_2/valid/`

* `data_clean_train_out`  
Output of train data after clean.  
Ex: `/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/30k/data_2/train/`
    
* `data_clean_valid_out`  
Output of valid data after clean.  
Ex: `/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/30k/data_2_cleannn/valid/`

* `data_train_predict`  
Logit prediction of `data_clean_train` when using a model to predict.  
Ex: `./logs/clean_model_1_resnet34/dataset.predictions.data_2_train.logits.satge1.5.npy`

* `data_valid_predict`  
Logit prediction of `data_clean_valid` when using a model to predict.  
Ex: `./logs/clean_model_1_resnet34/dataset.predictions.data_2_valid.logits.satge1.5.npy` 

# Make submission  
```python
python make_submission.py
```
Make sure you change correct `log_dir` in `make_submission.py`

# How to resume  
Define the `resume` in `train.yml` and `Run model` again. Usually, we will resume from `checkpoint.best.pth.tar`
in the `logs` folder.

### Supported architectures and models

#### From [torchvision](https://github.com/pytorch/vision/) package:

- ResNet (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`)
- DenseNet (`densenet121`, `densenet169`, `densenet201`, `densenet161`)
- Inception v3 (`inception_v3`)
- VGG (`vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`)
- SqueezeNet (`squeezenet1_0`, `squeezenet1_1`)
- AlexNet (`alexnet`)

#### From [Pretrained models for PyTorch](https://github.com/Cadene/pretrained-models.pytorch) package:
- ResNeXt (`resnext101_32x4d`, `resnext101_64x4d`)
- NASNet-A Large (`nasnetalarge`)
- NASNet-A Mobile (`nasnetamobile`)
- Inception-ResNet v2 (`inceptionresnetv2`)
- Dual Path Networks (`dpn68`, `dpn68b`, `dpn92`, `dpn98`, `dpn131`, `dpn107`)
- Inception v4 (`inception_v4`)
- Xception (`xception`)
- Squeeze-and-Excitation Networks (`senet154`, `se_resnet50`, `se_resnet101`, `se_resnet152`, `se_resnext50_32x4d`, `se_resnext101_32x4d`)
- PNASNet-5-Large (`pnasnet5large`)
- PolyNet (`polynet`)