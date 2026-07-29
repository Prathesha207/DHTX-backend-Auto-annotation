import re

with open('app/services/ml_runner.py', 'r', encoding='utf-8') as f:
    text = f.read()

if 'from datetime import datetime' not in text:
    text = text.replace('import sys', 'import sys\nfrom datetime import datetime')

with open('app/services/ml_runner.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("ml_runner patched")
