# Off-workspace storage (optional, off by default)

Kivra checkpoints and datasets can be large. `kivra/storage/connectors.py`
provides optional connectors for **Supabase Storage**, **Dropbox**, and **Google
Drive** so you can keep big artifacts off your local disk.

## Nothing is uploaded automatically

Connectors are invoked only when you explicitly call them, and only after
setting credentials. Local-only is the default.

## Configure (example `.env`)

```bash
cp .env.example .env
# edit .env with ONE backend's credentials
source .env  # or use direnv / your preferred loader
```

Environment variables:

| Backend | Required vars |
|---|---|
| supabase | `KIVRA_STORAGE_BACKEND=supabase`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `SUPABASE_BUCKET` |
| dropbox | `KIVRA_STORAGE_BACKEND=dropbox`, `DROPBOX_REFRESH_TOKEN`, `DROPBOX_APP_KEY`, `DROPBOX_APP_SECRET` (or `DROPBOX_ACCESS_TOKEN`) |
| gdrive | `KIVRA_STORAGE_BACKEND=gdrive`, `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_DRIVE_FOLDER_ID` |

## Use

```python
from kivra.storage.connectors import StorageConfig, get_connector

cfg = StorageConfig(
    backend="supabase",
    bucket="kivra-checkpoints",
    prefix="kivra/spark/",
)
conn = get_connector(cfg)
conn.upload_file("runs/kivra-spark/adapter_model.safetensors", "adapter_model.safetensors")
conn.download_file("adapter_model.safetensors", "restored.safetensors")
print(conn.list_files("kivra/spark/"))
```

Install optional SDKs: `pip install -e ".[storage]"`.

## Free-tier notes

- Supabase free tier includes 1 GB storage — fine for adapter checkpoints, not full 14B models.
- Dropbox Basic: 2 GB free.
- Google Drive: 15 GB free shared across Drive/Gmail/Photos.

These can hold adapters and small GGUF exports; full base checkpoints (several GB each) are best re-downloaded from Hugging Face rather than mirrored, to respect redistribution terms and keep provenance clear.
