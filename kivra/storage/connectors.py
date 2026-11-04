# Optional connectors: Supabase Storage, Dropbox, Google Drive.
#
# Each connector is a thin wrapper around its SDK. Importing this module does not
# require any SDK; import errors surface only when a backend is selected and its
# package/credentials are missing. Nothing is uploaded automatically.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StorageConfig:
    backend: str  # "supabase" | "dropbox" | "gdrive"
    bucket: str | None = None
    prefix: str = "kivra/"


class StorageConnector(ABC):
    @abstractmethod
    def upload_file(self, local_path: str, remote_name: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def download_file(self, remote_name: str, local_path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def list_files(self, prefix: str = "") -> list[str]:
        raise NotImplementedError


class NotConfigured(RuntimeError):
    pass


class SupabaseConnector(StorageConnector):
    def __init__(self, bucket: str, prefix: str = "kivra/"):
        try:
            from supabase import create_client  # type: ignore
        except ImportError as exc:
            raise NotConfigured(
                "supabase-py is not installed: pip install -e '.[storage]'"
            ) from exc
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise NotConfigured("Set SUPABASE_URL and SUPABASE_SERVICE_KEY")
        self.client = create_client(url, key)
        self.bucket = bucket
        self.prefix = prefix

    def _key(self, remote_name: str) -> str:
        return self.prefix + remote_name

    def upload_file(self, local_path: str, remote_name: str) -> str:
        with open(local_path, "rb") as fh:
            res = self.client.storage.from_(self.bucket).upload(
                self._key(remote_name), fh, {"upsert": "true"}
            )
        return res

    def download_file(self, remote_name: str, local_path: str) -> str:
        data = self.client.storage.from_(self.bucket).download(self._key(remote_name))
        with open(local_path, "wb") as fh:
            fh.write(data)
        return local_path

    def list_files(self, prefix: str = "") -> list[str]:
        files = self.client.storage.from_(self.bucket).list(self.prefix + prefix)
        return [f["name"] for f in files]


class DropboxConnector(StorageConnector):
    def __init__(self, prefix: str = "kivra/"):
        try:
            import dropbox  # type: ignore
        except ImportError as exc:
            raise NotConfigured(
                "dropbox-sdk is not installed: pip install -e '.[storage]'"
            ) from exc
        token = os.environ.get("DROPBOX_REFRESH_TOKEN")
        app_key = os.environ.get("DROPBOX_APP_KEY")
        app_secret = os.environ.get("DROPBOX_APP_SECRET")
        if not (token or os.environ.get("DROPBOX_ACCESS_TOKEN")):
            raise NotConfigured(
                "Set DROPBOX_REFRESH_TOKEN (+ app key/secret) or DROPBOX_ACCESS_TOKEN"
            )
        self.dbx = dropbox.Dropbox(
            oauth2_refresh_token=token,
            app_key=app_key,
            app_secret=app_secret,
        )
        self.prefix = prefix

    def _path(self, remote_name: str) -> str:
        return self.prefix + remote_name

    def upload_file(self, local_path: str, remote_name: str) -> str:
        from dropbox.files import WriteMode  # type: ignore

        with open(local_path, "rb") as fh:
            return self.dbx.files_upload(
                fh.read(), self._path(remote_name), mode=WriteMode.overwrite
            )

    def download_file(self, remote_name: str, local_path: str) -> str:
        _, res = self.dbx.files_download(self._path(remote_name))
        with open(local_path, "wb") as fh:
            fh.write(res.content)
        return local_path

    def list_files(self, prefix: str = "") -> list[str]:
        result = self.dbx.files_list_folder(self.prefix + prefix)
        return [e.name for e in result.entries]


class GDriveConnector(StorageConnector):
    def __init__(self, folder_id: str | None = None, prefix: str = "kivra/"):
        try:
            from google.oauth2 import service_account  # type: ignore
            from googleapiclient.discovery import build  # type: ignore
        except ImportError as exc:
            raise NotConfigured(
                "google-api-python-client/google-auth not installed: pip install -e '.[storage]'"
            ) from exc
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path:
            raise NotConfigured("Set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON")
        creds = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        self.service = build("drive", "v3", credentials=creds)
        self.folder_id = folder_id or os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
        self.prefix = prefix

    def _query(self, name: str) -> str:
        parts = ["name = '" + name + "'", "trashed = false"]
        if self.folder_id:
            parts.append("'" + self.folder_id + "' in parents")
        return " and ".join(parts)

    def upload_file(self, local_path: str, remote_name: str) -> str:
        from googleapiclient.http import MediaFileUpload  # type: ignore

        metadata = {"name": remote_name}
        if self.folder_id:
            metadata["parents"] = [self.folder_id]
        matches = self.service.files().list(q=self._query(remote_name)).execute().get("files", [])
        media = MediaFileUpload(local_path, resumable=True)
        if matches:
            return self.service.files().update(fileId=matches[0]["id"], media_body=media).execute()
        return self.service.files().create(body=metadata, media_body=media).execute()

    def download_file(self, remote_name: str, local_path: str) -> str:
        files = self.service.files().list(q=self._query(remote_name)).execute().get("files", [])
        if not files:
            raise FileNotFoundError(remote_name + " not found in Drive")
        request = self.service.files().get_media(fileId=files[0]["id"])
        with open(local_path, "wb") as fh:
            fh.write(request.execute())
        return local_path

    def list_files(self, prefix: str = "") -> list[str]:
        files = self.service.files().list(q="trashed = false").execute().get("files", [])
        return [f["name"] for f in files]


def get_connector(config: StorageConfig) -> StorageConnector:
    if config.backend == "supabase":
        return SupabaseConnector(bucket=config.bucket or "kivra-checkpoints", prefix=config.prefix)
    if config.backend == "dropbox":
        return DropboxConnector(prefix=config.prefix)
    if config.backend == "gdrive":
        return GDriveConnector(
            folder_id=os.environ.get("GOOGLE_DRIVE_FOLDER_ID"), prefix=config.prefix
        )
    raise ValueError("Unknown storage backend: " + config.backend)
