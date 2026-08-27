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

Use a launcher. It asks `port_resolver.py` for the port you set in Settings and
passes it to uvicorn, so the server and the settings screen agree:

```bash
./start_backend.sh          # or start_backend.ps1 / start_backend.bat
```

If you want auto-reload, carry the port yourself. A bare
`uvicorn app.main:app --reload` binds **8000 whatever your setting says**, and
then the app reports one port while the socket is on another:

```bash
uvicorn app.main:app --reload --port "$(python port_resolver.py)"
```

```powershell
uvicorn app.main:app --reload --port (python port_resolver.py)
```

## Available Endpoints

- `GET /plugins`: List all available training plugins.
- `GET /plugins/{id}/schema`: Get the JSON schema for a specific plugin's configuration.
