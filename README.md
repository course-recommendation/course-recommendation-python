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

## Start app
`uvicorn api.mod:app`