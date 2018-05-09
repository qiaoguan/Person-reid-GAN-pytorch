# DCGAN in Tensorflow

## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12) (Notice that it is not the latest version)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)
- CUDA 8.0

Add Cuda Path to bashrc first
```bash
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH"
```

We recommend you to install anaconda. Here we write a simple script for you to install the dependence by anaconda.
```python
# install env (especially for old version Tensorflow)
conda env create -f dcgan.yml
# activate env, then you can run code in this env without downgrading the outside Tensorflow.
source activate dcgan
```

### Let's start

### 1.Train
```bash
mkdir data
ln -rs your_dataset_path/DukeMTMC-reID/bounding_box_train ./data/duke_train
python main.py --dataset duke_train --train --input_height 128 --output_height 128 --options 1
```
`duke_train` is the dir path which contains images. Here I use the (DukeMTMC-reID)[https://github.com/layumi/DukeMTMC-reID_evaluation] training set. You can change it to your dataset path.

### 2.Test
```bash
python main.py --dataset duke_train --options 5  --output_path duke_256_48000  --sample_size 48000  --input_height 128 --output_height 128
```
It will use your trained model and generate 48000 images for the following semi-supervised training.
