import os
import winreg
from pathlib import Path

def get_windows_desktop() -> Path:
    """Dynamically get the actual Windows Desktop, handling OneDrive redirection."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as key:
            desktop_path, _ = winreg.QueryValueEx(key, "Desktop")
            return Path(os.path.expandvars(desktop_path))
    except Exception:
        return Path.home() / "Desktop"

def get_writable_default_location() -> str | None:
    desktop = get_windows_desktop()
    
    dhtx_dir = desktop / "DHTX Inspection Data"
    try:
        dhtx_dir.mkdir(parents=True, exist_ok=True)
        test_file = dhtx_dir / ".test_write"
        test_file.touch()
        test_file.unlink()
        return str(dhtx_dir)
    except Exception:
        return None
