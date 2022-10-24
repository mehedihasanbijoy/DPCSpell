<h1 align="center">DCSpell</h1>

## Activate the Environment
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

### Corrector Network

```
python corrector.py --HID_DIM 128 --ENC_LAYERS 5 --DEC_LAYERS 5 --ENC_HEADS 8 --DEC_HEADS 8 --ENC_PF_DIM 256 --DEC_PF_DIM 256 --ENC_DROPOUT 0.1 --DEC_DROPOUT 0.1 --CLIP 1 --LEARNING_RATE 0.0005 --N_EPOCHS 100 
```
