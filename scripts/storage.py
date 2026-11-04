"""Kivra storage CLI — offload datasets & checkpoints to free cloud storage.

Usage:
  python scripts/storage.py --backend supabase --bucket kivra-checkpoints upload runs/kivra-spark
  python scripts/storage.py --backend dropbox download runs/kivra-spark remote/kivra-spark
  python scripts/storage.py --backend gdrive --folder-id <DRIVE_FOLDER_ID> list

Free tiers (approx., verify with provider):
  supabase : 1 GB  (good for LoRA adapters / small GGUF)
  dropbox  : 2 GB
  gdrive   : 15 GB (shared across Drive/Gmail/Photos)

Nothing is uploaded unless you run an upload command explicitly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from kivra.storage.connectors import (
    StorageConfig,
    get_connector,
)

BACKENDS = ("supabase", "dropbox", "gdrive")

FREE_TIERS = {
    "supabase": "1 GB",
    "dropbox": "2 GB",
    "gdrive": "15 GB (shared with Gmail/Photos)",
}

ENV_VARS = {
    "supabase": ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"],
    "dropbox": ["DROPBOX_REFRESH_TOKEN", "DROPBOX_APP_KEY", "DROPBOX_APP_SECRET"],
    "gdrive": ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_DRIVE_FOLDER_ID"],
}


def _env_ok(backend: str) -> list[str]:
    missing = [v for v in ENV_VARS[backend] if not os.environ.get(v)]
    return missing


def cmd_list(args: argparse.Namespace) -> int:
    conn = get_connector(StorageConfig(backend=args.backend, bucket=args.bucket))
    prefix = args.prefix or "kivra/"
    files = conn.list_files(prefix)
    print(f"Files in {args.backend} under '{prefix}':")
    if not files:
        print("  (empty)")
    for f in files:
        print(f"  {f}")
    return 0


def _iter_local_files(local_dir: str, remote_prefix: str):
    root = Path(local_dir)
    for p in root.rglob("*"):
        if p.is_file():
            remote = str(Path(remote_prefix) / p.relative_to(root)).replace("\\", "/")
            yield str(p), remote


def cmd_upload(args: argparse.Namespace) -> int:
    missing = _env_ok(args.backend)
    if missing:
        print(f"Missing env vars for {args.backend}: {', '.join(missing)}")
        print("Copy .env.example to .env and fill them in, then re-run.")
        return 1

    conn = get_connector(
        StorageConfig(backend=args.backend, bucket=args.bucket, prefix=args.prefix)
    )
    local = Path(args.local)
    if not local.exists():
        print(f"Local path not found: {args.local}")
        return 1

    count = 0
    if local.is_file():
        files = [(str(local), args.remote or local.name)]
    else:
        files = list(_iter_local_files(str(local), args.remote or "kivra/"))

    for local_path, remote_name in files:
        print(f"Uploading {local_path} -> {args.backend}:{remote_name}")
        try:
            conn.upload_file(local_path, remote_name)
        except Exception as exc:  # pragma: no cover - network dependent  # noqa: BLE001
            print(f"  FAILED: {exc}")
            return 1
        count += 1
    print(f"Uploaded {count} file(s).")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    missing = _env_ok(args.backend)
    if missing:
        print(f"Missing env vars for {args.backend}: {', '.join(missing)}")
        return 1

    conn = get_connector(
        StorageConfig(backend=args.backend, bucket=args.bucket, prefix=args.prefix)
    )
    print(f"Downloading {args.remote} from {args.backend} -> {args.local}")
    try:
        conn.download_file(args.remote, args.local)
    except Exception as exc:  # pragma: no cover - network dependent  # noqa: BLE001
        print(f"  FAILED: {exc}")
        return 1
    print("Download complete.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Kivra cloud-storage helper (Supabase / Dropbox / Google Drive)"
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        required=True,
        help="Storage backend to use",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Bucket name (supabase only; default: kivra-checkpoints)",
    )
    parser.add_argument(
        "--folder-id",
        default=None,
        help="Drive folder ID (gdrive only; uses GOOGLE_DRIVE_FOLDER_ID if unset)",
    )
    parser.add_argument(
        "--prefix",
        default="kivra/",
        help="Remote key prefix (default: kivra/)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List files in a remote prefix")
    p_list.set_defaults(func=cmd_list)

    p_up = sub.add_parser("upload", help="Upload a file or directory")
    p_up.add_argument("local", help="Local file or directory to upload")
    p_up.add_argument("remote", nargs="?", default=None, help="Remote name/prefix")
    p_up.set_defaults(func=cmd_upload)

    p_down = sub.add_parser("download", help="Download a remote file")
    p_down.add_argument("remote", help="Remote file name")
    p_down.add_argument("local", help="Local destination path")
    p_down.set_defaults(func=cmd_download)

    args = parser.parse_args(argv)

    if args.backend == "gdrive" and args.folder_id:
        os.environ["GOOGLE_DRIVE_FOLDER_ID"] = args.folder_id

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
