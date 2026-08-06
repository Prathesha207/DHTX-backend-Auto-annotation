import json
import csv
import os
import datetime

class ReportGenerator:
    def __init__(self, reports_dir="backend/tests/reports"):
        self.reports_dir = os.path.abspath(reports_dir)
        os.makedirs(self.reports_dir, exist_ok=True)
        self.timeline_events = []
        self.metrics = []
        self.failures = []

    def record_event(self, event_name, details=""):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.timeline_events.append({"time": now, "event": event_name, "details": details})

    def record_metric(self, metric_name, value, unit="", threshold_pass=None, threshold_fail=None):
        status = "PASS"
        if threshold_fail is not None and value >= threshold_fail:
            status = "FAIL"
        elif threshold_pass is not None and value > threshold_pass:
            status = "WARN"
            
        self.metrics.append({
            "name": metric_name,
            "value": value,
            "unit": unit,
            "status": status
        })

    def record_failure(self, test_name, error_msg):
        self.failures.append({"test": test_name, "error": error_msg})

    def generate_timeline_csv(self):
        path = os.path.join(self.reports_dir, "timeline.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "event", "details"])
            writer.writeheader()
            writer.writerows(self.timeline_events)

    def generate_metrics_csv(self):
        path = os.path.join(self.reports_dir, "metrics.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "value", "unit", "status"])
            writer.writeheader()
            writer.writerows(self.metrics)

    def generate_failures_csv(self):
        path = os.path.join(self.reports_dir, "failures.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["test", "error"])
            writer.writeheader()
            writer.writerows(self.failures)

    def generate_html_report(self):
        # A simple HTML template
        html = f"""
        <html>
        <head><title>Stress Report</title></head>
        <body>
        <h1>Production Verification Report</h1>
        <p>Generated at: {datetime.datetime.now().isoformat()}</p>
        <h2>Metrics</h2>
        <table border="1">
        <tr><th>Name</th><th>Value</th><th>Unit</th><th>Status</th></tr>
        """
        for m in self.metrics:
            html += f"<tr><td>{m['name']}</td><td>{m['value']}</td><td>{m['unit']}</td><td>{m['status']}</td></tr>\n"
        html += "</table><h2>Failures</h2><ul>"
        for f in self.failures:
            html += f"<li><b>{f['test']}:</b> {f['error']}</li>"
        html += "</ul></body></html>"
        
        path = os.path.join(self.reports_dir, "stress_report.html")
        with open(path, "w") as f:
            f.write(html)
            
    def generate_all(self):
        self.generate_timeline_csv()
        self.generate_metrics_csv()
        self.generate_failures_csv()
        self.generate_html_report()
