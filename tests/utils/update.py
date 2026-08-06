import re

file_path = 'd:\\Dhtx-auto-annotation\\frontend\\src\\types\\inference.ts'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('id: string;\n  level: ''info''', 'id: string | number;\n  level: ''info''')
content = content.replace('id: number;\n  level: ''info''', 'id: string | number;\n  level: ''info''')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated LiveLog id type to string | number")
