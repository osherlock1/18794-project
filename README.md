
## Instructions

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You might need to run
```
pip install -e .
```


### Download dataset
make sure to set up your kaggle key

```
python scripts/download_data.py
unzip data/asl-signs.zip
```

### Set up data for training
```
python scripts/process_data.py
```

this will create too files in ```data/asl-signs/cleaned```
```
sequences_noface.npy
labels.npy
```
### Training
This script is set up to work with a config to keep tracking on our hyper parameters and etc.  It is automatically set up with weights and biases (wandb)

```
python scripts/run_training.py --config "config name"
```
