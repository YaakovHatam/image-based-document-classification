

# Image-Based Document Classification

## Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
pip install -r .\requirements.txt
```


## Run Main Script

To run the main program:

```powershell
python .\main.py
```

## Run Tests

To execute tests using pytest:

```powershell
pytest -q
```

## Build

This project can be packaged into a redistributable bundle (single-file executable + config files + templates) using the included build scripts.

It will create a new folder containing:
    1. The built executable
    2. `config.toml` (copied as-is)
    3. `templates_pdf` directory (copied as-is)
    4. Empty `to_process` directory (created)

**Windows:**
```powershell
./build.bat
```
\
**Linux:**
```powershell
chmod +x build.sh
./build.sh
```



