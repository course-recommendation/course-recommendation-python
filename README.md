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

## Set environment variables
Fill in the .env file following `.env.example`.

## Start app
```
uvicorn api.mod:app
```

## Build and upload trirank to Azure Blob Storage (if not already done)
```
python3 trirank.py
python3 upload_trirank.py
```
