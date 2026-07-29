import re

with open('models/inference_video_full_detection_new.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        if Path(final_path).exists():
            Path(final_path).unlink()
        shutil.move(self.temp_path, final_path)'''

replacement = '''        if Path(final_path).exists():
            Path(final_path).unlink()
            
        # Retry loop for WinError 32 (file in use by VideoWriter async release)
        import time
        max_retries = 10
        for attempt in range(max_retries):
            try:
                shutil.move(self.temp_path, final_path)
                break
            except PermissionError:
                if attempt == max_retries - 1:
                    print(f"[WARN] Could not move {self.temp_path} after {max_retries} retries.")
                time.sleep(0.1)
            except OSError:
                break'''

text = text.replace(target, replacement)

with open('models/inference_video_full_detection_new.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("shutil.move patched")
