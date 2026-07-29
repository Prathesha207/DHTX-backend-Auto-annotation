import re

with open('app/services/inference_state_machine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the start of the while loop
loop_start_pattern = r'(\s+)(# .*Frame loop.*\n\s+)while True:'
match_start = re.search(loop_start_pattern, content)
if not match_start:
    print("Could not find start of loop")
    exit(1)

# Find the end of the loop / start of cleanup
loop_end_pattern = r'(\s+)(# .*End of video.*\n\s+# .*mid-cycle.*\n\s+if self\.cycle_mgr\.active:\n\s+self\._finalize_current_cycle\(\)\n\n\s+cap\.release\(\))'
match_end = re.search(loop_end_pattern, content)
if not match_end:
    print("Could not find end of loop")
    exit(1)

indent = match_start.group(1)
start_idx = match_start.start(2)
end_idx = match_end.start(2)

# Extract the loop block and indent it
loop_block = content[start_idx:end_idx]
indented_loop = ""
for line in loop_block.split('\n'):
    if line:
        indented_loop += "    " + line + "\n"
    else:
        indented_loop += "\n"

# The cleanup block
cleanup_block = content[end_idx:match_end.end(2)]
cleanup_replaced = cleanup_block.replace("if self.cycle_mgr.active:", "if self.cycle_mgr and self.cycle_mgr.active:")
cleanup_replaced = cleanup_replaced.replace("self._finalize_current_cycle()", "self._finalize_current_cycle(abort=True)")
cleanup_replaced = cleanup_replaced.replace("cap.release()", "if cap:\n            cap.release()")

indented_cleanup = ""
for line in cleanup_replaced.split('\n'):
    if line:
        indented_cleanup += "    " + line + "\n"
    else:
        indented_cleanup += "\n"

new_block = "try:\n" + indented_loop + indent + "finally:\n" + indented_cleanup

# Replace in content
new_content = content[:start_idx] + new_block + content[match_end.end(2):]

with open('app/services/inference_state_machine.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Replaced successfully")
