<h1 align="center">DPCSpell</h1>
<h2 align="center">
  A Transformer-based Detector-Purificator-Corrector Framework for Spelling Error Correction of Bangla and Resource Scarce Indic Languages    
</h2>

![dpcspell](https://user-images.githubusercontent.com/58245357/194469283-c7dbfc0b-391e-4214-a6a2-99b7ba2dc512.png)


## Running Test
| Operating System  | Requirement | Remark |
| ------------- | ------------- | ------------- |
| Ubuntu 16.04.7 LTS  | requirements_ubuntu.txt  | :heavy_check_mark: Successful |
| Ubuntu 18.04.6 LTS (Google Colab)  | requirements_colab.txt  | - |
| Windows 10  | requirements_windows.txt  | - |


## Detector Network

```
python detector.py --HID_DIM 128 --ENC_LAYERS 5 --DEC_LAYERS 5 --ENC_HEADS 8 --DEC_HEADS 8 --ENC_PF_DIM 256 --DEC_PF_DIM 256 --ENC_DROPOUT 0.1 --DEC_DROPOUT 0.1 --CLIP 1 --LEARNING_RATE 0.0005 --N_EPOCHS 100 
```

## Purificator Network

```
python purificator.py --HID_DIM 128 --ENC_LAYERS 5 --DEC_LAYERS 5 --ENC_HEADS 8 --DEC_HEADS 8 --ENC_PF_DIM 256 --DEC_PF_DIM 256 --ENC_DROPOUT 0.1 --DEC_DROPOUT 0.1 --CLIP 1 --LEARNING_RATE 0.0005 --N_EPOCHS 100 
```

## Corrector Network

```

```
