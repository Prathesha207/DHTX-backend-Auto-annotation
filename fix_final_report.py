import os

with open('app/services/inference_state_machine.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_code = '''        # Final report
        report = self.cycle_mgr.final_report()
        self._log_info(
            f"FINAL REPORT: total={report['total_cycles']} "
            f"passed={report['passed']} "
            f"failed={report['failed']} "
            f"unknown={report['unknown']}"
        )'''

new_code = '''        # Final report
        self.cycle_mgr.final_report()
        self._log_info(
            f"FINAL REPORT: total={self.cycle_mgr.total_cycles} "
            f"passed={self.cycle_mgr.passed} "
            f"failed={self.cycle_mgr.failed} "
            f"unknown={self.cycle_mgr.unknown}"
        )'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('app/services/inference_state_machine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed final_report usage.")
else:
    print("Old code block not found.")
