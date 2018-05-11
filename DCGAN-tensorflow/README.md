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
copy your dataset(market) to data folder
python main.py --dataset market --options 1
```
`market` is the dir path which contains images. You can change it to your dataset path.

### 2.Test
```bash
python main.py --dataset market --options 5  --output_path gen_market  --sample_size 24000
python resizeImage.py
```
It will use your trained model and generate 24000 images for the following semi-supervised training. the generated images are stored in gen_market folder,  then run resizeImage.py,  the generated images will be resized into 126*64 stored in gen_0000 market.  after that, add gen_0000 to the training set of market, so the generated images will be used as training image to help model training.
