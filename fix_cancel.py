import os
import sys

with open('app/services/inference_state_machine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace signature
content = content.replace('def run(self):', 'def run(self, cancel_event=None):')

# Add check inside the loop
old_loop_start = '        try:\n    # -- Frame loop -------------------------------------------\n            while True:\n                ret, frame = cap.read()'
new_loop_start = '        try:\n    # -- Frame loop -------------------------------------------\n            while True:\n                if cancel_event and cancel_event.is_set():\n                    self._log_warning("Inference cancelled via cancel_event")\n                    break\n                ret, frame = cap.read()'

# Since the previous replace didn't work exactly as expected, let's find the while True: loop inside the try block
if 'while True:' in content and 'try:' in content:
    content = content.replace(
        '            while True:\n                ret, frame = cap.read()',
        '            while True:\n                if cancel_event and cancel_event.is_set():\n                    self._log_warning("Inference cancelled via cancel_event")\n                    break\n                ret, frame = cap.read()'
    )

with open('app/services/inference_state_machine.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated run() signature and loop.")
