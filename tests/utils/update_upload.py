import re

file_path = 'd:\\Dhtx-auto-annotation\\backend\\app\\services\\upload_service.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

pat = re.compile(r'file_uuid = uuid4\(\)\.hex\s*\n\s*filename = f"\{file_uuid\}\{extension\}"')
new = '''file_uuid = uuid4().hex
        filename = f"{sanitized_name}{extension}"'''

content = pat.sub(new, content, count=1)
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated upload_service.py")
