# 关于这个项目

我找了GitHub上不少仓库，也看了不少issue，用了两天的时间，发现大部分代码几乎都过时了，甚至没法跑 :(
我改了一些代码和依赖，至少现在他能跑了 :)

下面有更多关于此项目的信息（有修改）


# PassGAN

这是一个[brannondorsey's implementation](https://github.com/brannondorsey/PassGAN)  [PassGAN](https://arxiv.org/abs/1709.00440)的分支 PassGAN是一个能够生成密码的GAN网络，但是他的代码有些过时了，我进行了一些更改
至少现在，2020，他能跑了.  [原版本的README](README.md.old)上面有更多原项目的信息.

## 修改

我只对原项目分支的分支进行了一些修改，让他能运行（1.15.0目前被弃用了）

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
