
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

This script reads through all of the parquet files and saves them in the format of 

64 frames x 1086 (x and y points)


### Files
```
src/
    data.py - data processing
    datasets.py - torch dataset classes
    models.py - torch models
    training.py - module for training related functions
    visualization.py - module for visualization related functions
```

