# The project

I tried many Repository, but most of the code is out date. Some of them even can't run :(
I edit some code and change the version of the Dependency. Now it can run :)

Here are the info of the project(edited)

[中文版(more)](CN_README.md)

# PassGAN

This is a fork of [brannondorsey's implementation](https://github.com/brannondorsey/PassGAN) of [PassGAN](https://arxiv.org/abs/1709.00440), a generative adversarial network built to make passwords. The repo was slightly out of date, relying on CUDA 8.0 and Tensorflow 1.4, as well as missing some dependencies that made it easy to train and run in 2019. View [the original documentation at README.md.old](README.md.old) for more information on this GAN.

## Modifications

No modifications have been made to the underlying network implementation other than optimisation and refactoring for Python 3.x and TensorFlow 1.15.0, which makes it compatible with [tensorflow-rocm](https://rocm.github.io/tensorflow.html) and tensorflow-gpu.

## Installation & Prerequisites

To install a CPU-only PassGAN and basic dependencies, run:

````bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
````

You will then need to install the correct GPU acceleration package. For ROCm, use:

````bash
pip install tensorflow==1.14.0
````

For NVIDIA/CUDA, use:

````bash
pip install tensorflow-gpu==1.14.0
````

## Quick start

You can train your own model based upon your password corpus. The `pretrained` model has been deleted from this repository as it contains Python 2.x `pickle` data that is incompatible with Python 3. Training with large data sets will take quite some time with a reasonably fast GPU.

### Training

From the original `README.md.old`:

```bash
# train for 200000 iterations, saving checkpoints every 5000
# uses the default hyperparameters from the paper
python train.py --output-dir output --training-data data/train.txt
```

### Generation

This example uses a model you have trained to generate 1,000,000 passwords:

```bash
python sample.py \
	--input-dir $YOUR_OUTPUT \
	--checkpoint $YOUR_OUTPUT/checkpoints/$YOUR_CHECKPOINT.ckpt \
	--output gen_passwords.txt \
	--batch-size 1024 \
	--num-samples 1000000
```

## Caveats

* This PassGAN implementation hardcodes the use of the NCHW format (for CuDNN) in a lot of places in `tflib/`, which is not supported on CPU (where it should be NHWC instead). Until this is patched to change the data format for CPU, you will need a GPU for training.
* Loading datasets for training will now assume that you are using UTF-8 for your dataset when running with Python 3. Note that many older dictionaries, such as `rockyou.txt`, are in `ISO-8859-1`. You can use `iconv` to transform them (e.g. `iconv -f iso-8859-1 -t UTF-8 trainingdata/rockyou.txt > trainingdata/rockyou.utf8.txt`).
