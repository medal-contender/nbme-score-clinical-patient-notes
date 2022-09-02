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
- Young Min Paik (ympaik@hotmail.com)
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
## Task
This task is a PIPELINE to experiment with PLM(Pretrained Language Model), which is often used in NLP tasks, and the following models are tested
- RoBERTa
- DeBERTa
- Electra
- LUKE

In addition, we conducted experiments by modifying FC Layer of the each model.

<div align="center">
    <img src="https://user-images.githubusercontent.com/68190553/176803353-58a9e0db-349a-4cfb-a07f-a8daef6a1de7.png" width="60%"/>
</div>

### Baseline 1
Baseline 1 is a task that predicts all tokens in the input and processes only tokens above the threshold, similar to the NER task in BERT.<br/>
It is a task in which the model recognizes and predicts 'text' and 'text_feature' by organizing Dataset in order of [‘text’, 'text_feature’].

### Baseline 2, 3
Baseline 2, 3 is based on baseline 1, but similar to QA task, Dataset is organized in the order of ['text_feature’, 'text’].<br/>
baseline 2 is a Pytorch-based code and baseline 3 is Transformer-based code.


### Main Issues

- Dataset review
  1. typo problem
    - Many typos were found on a given dataset, and we solved these problems in the data preprocessing process.
  2. Related Issues
    - https://github.com/medal-contender/nbme-score-clinical-patient-notes/issues/76
    - https://github.com/medal-contender/nbme-score-clinical-patient-notes/issues/75
    - https://github.com/medal-contender/nbme-score-clinical-patient-notes/issues/74
    - https://github.com/medal-contender/nbme-score-clinical-patient-notes/issues/63
    - https://github.com/medal-contender/nbme-score-clinical-patient-notes/issues/52
- Data Embedding
  1. Tokenizer issue
    - In the process of embedding data, we found that each tokenizer calculates the position of the offset differently. To solve this problem, we have tried methods such as implementing token positioning functions, char-based embeddings, and we have been able to confirm good performance.
  2. Rlated Issues
    - https://github.com/medal-contender/nbme-score-clinical-patient-notes/issues/66
    - https://github.com/medal-contender/nbme-score-clinical-patient-notes/issues/44
    - https://github.com/medal-contender/nbme-score-clinical-patient-notes/issues/37
- Model performance
  1. Scheduler
    - We analyzed various studies and techniques related to schedulers in which the model can reach the global minimum, and applied the SOTA scheduler CosineAnnealingWarmupRestarts to improve model performance.
  2. Model tuning
    - To improve the performance of the model, I thought of a tuning method, and I improved the performance of the model by tuning the layer part.
  3. Ensemble Strategy
    - We discussed strategies for ensembling different models (Roberta, DeBERTa, etc.) because each model has a different structure and finally decided on three models to achieve the best performance.
  4. Related Issues
    - https://github.com/medal-contender/nbme-score-clinical-patient-notes/issues/30
    - https://github.com/medal-contender/nbme-score-clinical-patient-notes/issues/7
    - https://github.com/medal-contender/nbme-score-clinical-patient-notes/pull/40
    - https://github.com/medal-contender/nbme-score-clinical-patient-notes/issues/65
    - https://github.com/medal-contender/nbme-score-clinical-patient-notes/issues/80

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

You can set `launch.json` for vscode as follow:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "run_train.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/nbme",
            "args": ["--config-file", "../configs/deberta.yaml", "--train"]
        }
    ]
}
```

## Requirements

### Environment

- Python 3.7 (To match with Kaggle environment)
- Conda
- git
- git-lfs
- CUDA 11.3 + PyTorch 1.10.1

Pytorch version may vary depanding on your hardware configurations.

### Installation with virtual environment

```bash
git clone https://github.com/medal-contender/nbme-score-clinical-patient-notes.git
conda create -n nbme python=3.7
conda activate nbme
# PyTorch installation process may vary depending on your hardware
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
cd nbme-score-clinical-patient-notes
pip install -r requirements.txt
```

Run this in WSL (or WSL2)

```bash
./download_pretrained_models.sh
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
