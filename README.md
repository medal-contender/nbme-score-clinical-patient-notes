# Jigsaw Rate Severity Of Toxic Comments
---
<p align="center">
  <img src="./images/nbme.png" width=550>
</p>


---

## Members

```
- Jeongwon Kim (kimkim031@naver.com)
- Jaewoo Park (jerife@naver.com)
- Youngmin Paik (ympaik@hotmail.com)
- Hyeonhoon Lee (jackli0373@gmail.com)
```

---

## Competition Overview

### Problem
- Please Fill In
### Dataset
- Ruddit Dataset
- Jigsaw Rate Severity of Toxic Comments
- toxic-task
### Due Date
- Team Merge Deadline - 2022/01/31
- Submission Deadline - 2022/02/07

---

## Program

- Fetch Pretrained Models
```shell
$ sh ./download_pretrained_models.sh
```

- Train
```shell
$ cd /jigsaw-toxic-severity-rating/jigsaw_toxic_severity_rating
$ python3 run_train.py \
          "--config-file", "/jigsaw-toxic-severity-rating/configs/roberta.yaml", \
          "--train", \
          "--training-keyword", "roberta-test"
```

## Requirements

### Environment

* Python 3.7 (To match with Kaggle environment)
* Conda
* git
* git-lfs
* CUDA 11.3 + PyTorch 1.10.1

Pytorch version may vary depanding on your hardware configurations.


### Installation with virtual environment (Windows)

```bash
git clone https://github.com/medal-challenger/jigsaw-rate-severity-of-toxic-comments.git
conda create -n jigsaw python=3.8
activate jigsaw
# PyTorch installation process may vary depending on your hardware
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
cd jigsaw-rate-severity-of-toxic-comments
pip install -r requirements.txt
```

Run this in WSL (or WSL2)
```bash
./download_pretrained_models.sh
```

### Installation (MacOS)

```bash
```

### To update the code

```bash
$ git pull
```

If you have local changes, and it causes to abort `git pull`, one way to get around this is the following:

```bash
# removing the local changes
$ git stash
# update
$ git pull
# put the local changes back on top of the recent update
$ git stash pop
```
