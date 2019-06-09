# Automatic-Image-Enhancement
Link to our dataset: [MIT-Adobe FiveK Dataset](https://data.csail.mit.edu/graphics/fivek/)

Download the data and preprocess it into patches before running the code.

## 1. Run Baseline Model

```bash
python style_transfer.py
```

(Default photo: a0001.tif, can change manually)

## 2. Run DCGAN

#### Train

```bash
python main.py --train --model_type DCGAN
```

#### Test image patches

```bash
python main.py --test_batches --resume_iter <resume_iter> --model_type DCGAN 
```

#### Test full images

```bash
python main.py --test --resume_iter <resume_iter> --model_type DCGAN 
```

## 3. Run WGAN

#### Train

```bash
python main.py --train --model_type WGAN
```

#### Test image patches

```bash
python main.py --test_batches --resume_iter <resume_iter> --model_type WGAN 
```

#### Test full images

```bash
python main.py --test --resume_iter <resume_iter> --model_type WGAN 
```