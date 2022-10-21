<h1 align="center">GRUSeq2Seq</h1>

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

## Training and Evaluation of GRUSeq2Seq
```
python main.py --CORPUS "./Dataset/corpus.csv" --ENC_EMB_DIM 128 --DEC_EMB_DIM 128 --ENC_HIDDEN_DIM 256 --DEC_HIDDEN_DIM 512 --ENC_DROPOUT 0.1 --DEC_DROPOUT 0.1 --MAX_LEN 48 --CLIP 1 --BATCH_SIZE 256 --LEARNING_RATE 0.0005 --N_EPOCHS 100
```
