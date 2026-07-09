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

## Login to Azure
```
az login
```

## Build and upload trirank to Azure Blob Storage (optional)
```
python3 trirank.py
python3 upload_trirank.py
```

## Start app
```
python3 -m uvicorn api.mod:app --host 0.0.0.0
```
