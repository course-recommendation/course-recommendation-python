## Create and activate virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```

## Install requirements
```
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

## Build trirank
```
python3 trirank.py
```

## Set environment variables
```
export MODEL_PATH=<path_to_model> (e.g., /Users/hien/Works/course-recommendation-python/save_dir/TriRank/2026-04-06_00-45-53-786624.pkl)
export TRAIN_SET_PATH=<path_to_trainset> (e.g., /Users/hien/Works/course-recommendation-python/save_dir/TriRank/2026-04-06_00-45-53-786624.pkl.trainset)
```

## Start app
```
uvicorn api.mod:app
```