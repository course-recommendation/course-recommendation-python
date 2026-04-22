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
Fill in the .env file following `.env.example`.

## Start app
```
uvicorn api.mod:app
```