import ast
import json
import sys

def analyze(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    
    imports = []
    functions = []
    classes = []
    assignments = []
    
    for node in tree.body:
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.append(name.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
        elif isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            functions.append({"name": node.name, "args": args})
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if isinstance(node.value, ast.Constant):
                        assignments.append({"name": target.id, "value": node.value.value})
    
    return {
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "assignments": assignments
    }

old_data = analyze('backend/models/inference_video_full_detection.py')
new_data = analyze('backend/models/inference_video_full_detection_new.py')

with open('analysis_out.json', 'w') as f:
    json.dump({"old": old_data, "new": new_data}, f, indent=2)
print("Analysis complete")
