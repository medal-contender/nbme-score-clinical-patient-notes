# NBME - Score Clinical Patient Notes
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

In this competition, you’ll identify specific clinical concepts in patient notes. Specifically, you'll develop an automated method to map clinical concepts from an exam rubric (e.g., “diminished appetite”) to various ways in which these concepts are expressed in clinical patient notes written by medical students (e.g., “eating less,” “clothes fit looser”). Great solutions will be both accurate and reliable.

If successful, you'll help tackle the biggest practical barriers in patient note scoring, making the approach more transparent, interpretable, and easing the development and administration of such assessments. As a result, medical practitioners will be able to explore the full potential of patient notes to reveal information relevant to clinical skills assessment.

### Due Date
- Team Merge Deadline - April 26, 2022
- Submission Deadline - May 3, 2022

---

## Program

- Fetch Pretrained Models
```shell
$ sh ./download_pretrained_models.sh
```

- Train
```shell
$ cd /nbme-score-clinical-patient-notes/nbme
$ python3 run_train.py \
          "--config-file", "nbme-score-clinical-patient-notes/configs/deberta.yaml", \
          "--train", \
          "--training-keyword", "deberta-test"
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
