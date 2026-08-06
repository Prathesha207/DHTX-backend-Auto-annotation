import json
import os

with open('analysis_out.json', 'r') as f:
    data = json.load(f)

old = data['old']
new = data['new']

report = []
report.append("# Phase 1: Inference Model Compatibility Analysis")
report.append("\nThis report analyzes the changes between `inference_video_full_detection.py` (OLD) and `inference_video_full_detection_new.py` (NEW).")

# 1. Imports
report.append("\n## 1. Imports")
old_imports = set(old['imports'])
new_imports = set(new['imports'])
added_imports = new_imports - old_imports
removed_imports = old_imports - new_imports

if added_imports or removed_imports:
    report.append("### Changes Detected")
    if added_imports:
        report.append(f"- **New Dependencies**: `{', '.join(added_imports)}`")
    if removed_imports:
        report.append(f"- **Removed Dependencies**: `{', '.join(removed_imports)}`")
else:
    report.append("✅ **No changes in imports.**")

# 2. Models Loaded
report.append("\n## 2. Configuration & Models Loaded")
old_assigns = {a['name']: a['value'] for a in old['assignments'] if isinstance(a['value'], (int, float, str, bool))}
new_assigns = {a['name']: a['value'] for a in new['assignments'] if isinstance(a['value'], (int, float, str, bool))}

# Check for model path assignments (e.g. YOLO, weights, etc.)
report.append("\n### Config & Constants Comparison")
report.append("| Config | Old | New | Status |")
report.append("|--------|-----|-----|--------|")
for k in sorted(set(old_assigns.keys()).union(new_assigns.keys())):
    if k.isupper() or 'path' in k.lower() or 'model' in k.lower():
        v_old = old_assigns.get(k, 'N/A')
        v_new = new_assigns.get(k, 'N/A')
        if v_old != v_new:
            report.append(f"| {k} | {v_old} | {v_new} | ⚠️ Changed |")

# 3. Function Signatures
report.append("\n## 3. Function Signatures")
old_funcs = {f['name']: f['args'] for f in old['functions']}
new_funcs = {f['name']: f['args'] for f in new['functions']}

added_funcs = set(new_funcs.keys()) - set(old_funcs.keys())
removed_funcs = set(old_funcs.keys()) - set(new_funcs.keys())
changed_funcs = []
for f in set(old_funcs.keys()).intersection(new_funcs.keys()):
    if old_funcs[f] != new_funcs[f]:
        changed_funcs.append((f, old_funcs[f], new_funcs[f]))

if added_funcs:
    report.append(f"- **New Functions**: `{', '.join(added_funcs)}`")
if removed_funcs:
    report.append(f"- **Removed Functions**: `{', '.join(removed_funcs)}`")
if changed_funcs:
    report.append("- **Changed Signatures**:")
    for f, o_args, n_args in changed_funcs:
        report.append(f"  - `{f}`:\n    - Old: `({', '.join(o_args)})`\n    - New: `({', '.join(n_args)})`")

# Breaking Changes Summary
report.append("\n## 4. Breaking Changes")
if removed_funcs or changed_funcs:
    for f in removed_funcs:
        report.append(f"1. ❌ Function removed: `{f}`")
    for f, _, _ in changed_funcs:
        report.append(f"1. ❌ Function signature changed: `{f}`")
else:
    report.append("✅ No immediate API breaking changes detected in function signatures.")

# Integration Impact
report.append("\n## 5. Integration Impact")
report.append("Based on the code analysis:")
if changed_funcs and 'process_video' in [f[0] for f in changed_funcs]:
    report.append("- `app/services/ml_runner.py` / `application_controller.py`: ❌ **Major rewrite** (Core entry point changed)")
else:
    report.append("- `application_controller.py`: ⚠️ **Small change** (Check callback parameters)")

with open("compatibility_report.md", "w") as f:
    f.write("\n".join(report))

print("Report generated in compatibility_report.md")
