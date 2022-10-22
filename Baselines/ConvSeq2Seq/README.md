<h1 align="center">ConvSeq2Seq</h1>

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

## Training and Evaluation of ConvSeq2Seq
```
python main.py --CORPUS "./Dataset/corpus.csv" --EMB_DIM 128 --ENC_LAYERS 5 --DEC_LAYERS 5 --ENC_KERNEL_SIZE 3 --DEC_KERNEL_SIZE 3 --ENC_DROPOUT 0.2 --DEC_DROPOUT 0.2 --CLIP 0.1 --BATCH_SIZE 256 --LEARNING_RATE 0.0005 --N_EPOCHS 100
```
