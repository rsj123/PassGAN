# 关于这个项目

我找了GitHub上不少仓库，也看了不少issue，用了两天的时间，发现大部分代码几乎都过时了，甚至没法跑 :(
我改了一些代码和依赖，至少现在他能跑了 :)

下面有更多关于此项目的信息（有修改）

# PassGAN

这是一个[brannondorsey's implementation](https://github.com/brannondorsey/PassGAN)  [PassGAN](https://arxiv.org/abs/1709.00440)的分支 PassGAN是一个能够生成密码的GAN网络，但是他的代码有些过时了，我进行了一些更改
至少现在，2020，他能跑了.  [原版本的README](README.md.old)上面有更多原项目的信息.

## 修改

我只对原项目分支的分支进行了一些修改，让他能运行（1.15.0目前被弃用了）

## 安装 & 准备

使用如下命令以安装一个只使用CPU的PassGAN和基础依赖：

````bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
````

你需要安装 GPU 对应的加速包。对于 ROCm，使用如下命令：

````bash
pip install tensorflow==1.14.0
````

对于 NVIDIA/CUDA，使用如下命令：

````bash
pip install tensorflow-gpu==1.14.0
````

## 快速启动

你可以根据你的密码语料库来训练你自己的模型。`pretrained` 模型已从此存储库中删除，因为它含有与 Python3 不兼容的 Python2.x `pickle` 数据。使用大型数据集进行培训需要花费相当长的时。，当然，使用 GPU加速 速度相当快。 

### 训练

来自于原始的 `README.md.old`:

```bash
# train for 200000 iterations, saving checkpoints every 5000
# uses the default hyperparameters from the paper
python train.py --output-dir output --training-data data/train.txt
```

### 生成

此示例使用经过训练的模型生成 1000000 个密码： 

```bash
python sample.py \
	--input-dir $YOUR_OUTPUT \
	--checkpoint $YOUR_OUTPUT/checkpoints/$YOUR_CHECKPOINT.ckpt \
	--output gen_passwords.txt \
	--batch-size 1024 \
	--num-samples 1000000
```

## 注意事项

* PassGAN在实现 `tflib/` 中的很多硬编码的地方使用 NCHW 格式（用于 CuDNN ），这在 CPU 上是不受支持的（如果需要支持，则应使用 NHWC ）。在修改成 CPU 的数据格式之前，您需要 GPU 来进行训练。

* 现在，在使用Python 3运行数据集时，我们假定您使用 UTF-8 作为训练所需装载的数据集的编码。请注意，许多较旧的字典，如 `rockyou.txt` ，都使用的为 `ISO-8859-1` 编码。您可以使用 `iconv` 来转换它们（例如 `iconv -f iso-8859-1 -t UTF-8 trainingdata/rockyou.txt > trainingdata/rockyou.utf8.txt`）。
