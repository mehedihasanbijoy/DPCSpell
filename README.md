<h1 align="center">DPCSpell</h1>
<h2 align="center">
  A Transformer-based Detector-Purificator-Corrector Framework for Spelling Error Correction of Bangla and Resource Scarce Indic Languages    
</h2>


## Detector Network

```
python detector.py --HID_DIM 128 --ENC_LAYERS 3 --DEC_LAYERS 3 --ENC_HEADS 8 --DEC_HEADS 8 --ENC_PF_DIM 256 --DEC_PF_DIM 256 --ENC_DROPOUT 0.1 --DEC_DROPOUT 0.1 --CLIP 1 --N_EPOCHS 100 --LEARNING_RATE 0.0005
```
