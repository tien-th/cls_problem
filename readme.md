# Setup Instructions

## 1. Environment Setup

Run the following command to set up the environment:
```bash
bash ./set_up/env.sh
```

Activate the environment:
```bash
conda activate cell_cls
```

## 2. Download Data

Run the following command to download the required data:
```bash
sh ./set_up/download_data.sh
```

## 3. Test Training Script

Before running the full training, test the `train.py` script with the following configuration on line 23:

### Configuration
```python
# Configuration
TRAIN_INDICATOR_CSV = "split/train1.csv"
VAL_INDICATOR_CSV = "split/val1.csv"
```

Run the script:
```bash
python train.py
```

## 4. If No Errors Occur

Update the configuration in `train.py` as follows:
```python
TRAIN_INDICATOR_CSV = "split/train.csv"
VAL_INDICATOR_CSV = "split/val.csv"
```

## 5. Relax

Once the changes are made, go to sleep. 😊
