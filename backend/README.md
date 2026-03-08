# MRLN Arcane Tuner Backend

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API

```bash
uvicorn app.main:app --reload
```

## Available Endpoints

- `GET /plugins`: List all available training plugins.
- `GET /plugins/{id}/schema`: Get the JSON schema for a specific plugin's configuration.
