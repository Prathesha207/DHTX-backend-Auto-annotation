import os
import platform
from pathlib import Path

def get_windows_desktop() -> Path:
    """Dynamically get the actual Windows Desktop, handling OneDrive redirection."""
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as key:
                desktop_path, _ = winreg.QueryValueEx(key, "Desktop")
                resolved = Path(os.path.expandvars(desktop_path))
                if resolved.exists():
                    return resolved
        except Exception:
            pass
    return Path.home() / "Desktop"

def is_path_writable(path_str: str | Path | None) -> bool:
    """Check if the given directory path exists and is writable on this machine."""
    if not path_str:
        return False
    try:
        p = Path(path_str)
        p.mkdir(parents=True, exist_ok=True)
        test_file = p / ".write_test"
        test_file.touch()
        test_file.unlink()
        return True
    except Exception:
        return False

def get_writable_default_location() -> str:
    """
    Returns a guaranteed writable storage directory on whatever machine this backend runs on.
    1. Desktop/DHTX Inspection Data
    2. Local repo storage/
    3. User home directory
    """
    candidates = [
        get_windows_desktop() / "DHTX Inspection Data",
        Path(__file__).resolve().parent.parent / "storage",
        Path.home() / "DHTX Inspection Data",
    ]
    for cand in candidates:
        if is_path_writable(cand):
            return str(cand.resolve())
            
    # Absolute fallback
    fallback = Path("./storage").resolve()
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback)

def resolve_file_location(stored_path: str | None, active_root: str | None, filename: str | None, subfolder: str = "") -> str | None:
    """
    Resolves a file path across different machines.
    If the stored absolute path from a previous machine doesn't exist,
    searches in the active storage directory on the current machine.
    """
    if stored_path and Path(stored_path).is_file():
        return stored_path
    
    if active_root and filename:
        candidate = Path(active_root) / subfolder / filename if subfolder else Path(active_root) / filename
        if candidate.is_file():
            return str(candidate.resolve())
            
        # Recursive glob search for the filename inside active_root
        for found in Path(active_root).glob(f"**/{filename}"):
            if found.is_file():
                return str(found.resolve())
                
    return stored_path

