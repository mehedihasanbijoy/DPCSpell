<h1 align="center">DPCSpell</h1>
<p align="center">
  A Transformer-based Detector-Purificator-Corrector Framework for Spelling Error Correction of Bangla and Resource Scarce Indic Languages </br> Preprint — <a href="https://arxiv.org/abs/2211.03730" target="_blank">https://arxiv.org/abs/2211.03730</a>
</p>



<!-- ![dpcspell](https://user-images.githubusercontent.com/58245357/194469283-c7dbfc0b-391e-4214-a6a2-99b7ba2dc512.png) -->
<!-- ![DPCSpellGif2](https://user-images.githubusercontent.com/58245357/197951922-8859c491-0c8e-44b4-a8f0-4b774122a060.gif) -->
<!-- ![DPCSpellGif](https://user-images.githubusercontent.com/58245357/197949190-ebdcf496-98c3-4506-897e-b2ef9a4efc29.gif) -->

## 

## How DPCSpell works?
<!-- ![DPCSpellGif](https://user-images.githubusercontent.com/58245357/197949190-ebdcf496-98c3-4506-897e-b2ef9a4efc29.gif) -->
![dpcspell](https://user-images.githubusercontent.com/58245357/201864934-ebd28fa7-17e2-4482-9534-81c29e02bf3f.gif)

## Running Test
| Operating System  | Requirement | Remark |
| ------------- | ------------- | ------------- |
| Ubuntu 16.04.7 LTS  | requirements_u.yml  | :heavy_check_mark: Successful |
| Ubuntu 18.04.6 LTS (Google Colab)  | requirements_c.txt  | :heavy_check_mark: Successful |
| Windows 10  | requirements_w.yml  | :heavy_check_mark: Successful |

<br>

## Get Started

```
git clone https://github.com/mehedihasanbijoy/DPCSpell.git
```
or manually **download** and **extract** the github repository of DPCSpell.

<br>

## Environment Setup
### Create A Virtual Environment
```
conda env create -f requirements_u.yml (for Ubuntu 16.04.7 LTS)
or
conda env create -f requirements_w.yml (for Windows 10)
```
<!-- conda env create -f requirements_c.txt (for Ubuntu 18.04.6 LTS in Colab) -->

### Activate the Environment
```
conda activate DPCSpell
```

<br>

## Prepare SEC Corpora 
```
gdown https://drive.google.com/drive/folders/1vfCAqqXy0ZTL8cPKR-K5q5coBnNv2Zxf?usp=sharing -O ./Dataset --folder
```
<p>
or manually <b>download</b> the folder from <a href="https://drive.google.com/drive/folders/1vfCAqqXy0ZTL8cPKR-K5q5coBnNv2Zxf?usp=sharing" target="_blank">here</a> and keep the extracted files into <b>./Dataset/</b>
</p>

<br>

## Training and Evaluation of DPCSpell

### Detector Network

```
python detector.py --CORPUS "./Dataset/corpus.csv" --HID_DIM 128 --ENC_LAYERS 5 --DEC_LAYERS 5 --ENC_HEADS 8 --DEC_HEADS 8 --ENC_PF_DIM 256 --DEC_PF_DIM 256 --ENC_DROPOUT 0.1 --DEC_DROPOUT 0.1 --CLIP 1 --LEARNING_RATE 0.0005 --N_EPOCHS 100
```

### Purificator Network

```
python purificator.py --HID_DIM 128 --ENC_LAYERS 5 --DEC_LAYERS 5 --ENC_HEADS 8 --DEC_HEADS 8 --ENC_PF_DIM 256 --DEC_PF_DIM 256 --ENC_DROPOUT 0.1 --DEC_DROPOUT 0.1 --CLIP 1 --LEARNING_RATE 0.0005 --N_EPOCHS 100 
```

### Corrector Network

```
python corrector.py --HID_DIM 128 --ENC_LAYERS 5 --DEC_LAYERS 5 --ENC_HEADS 8 --DEC_HEADS 8 --ENC_PF_DIM 256 --DEC_PF_DIM 256 --ENC_DROPOUT 0.1 --DEC_DROPOUT 0.1 --CLIP 1 --LEARNING_RATE 0.0005 --N_EPOCHS 100 
```

<br>

## Benchmarking Bangla SEC Task

![benchmark](https://user-images.githubusercontent.com/58245357/195144459-0150f456-f06b-4aff-93f5-36b1fb76ea42.png)
