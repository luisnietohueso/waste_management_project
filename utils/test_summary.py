"""
Author: Luis Nieto Hueso
Date: 28/03/2025
Description:
This script generates a comprehensive markdown summary report for automated tests related to the waste management system.
It logs the status of each test (pass/fail) and includes any relevant plots to aid in result interpretation.

Function: `generate_summary_report()`
- Accepts a dictionary of test results and a list of plot file paths.
- Writes a markdown report that includes a timestamp, test outcomes, and embedded images for visual reference.
- Status icons (✅ / ❌) are used to clearly indicate whether each test passed or failed.
- The destination directory is created automatically if it does not exist.
- Designed for inclusion in CI/CD test pipelines or documentation outputs.

Main Block (when run as a standalone script):
- Defines example test results and a list of plot image paths.
- Calls `generate_summary_report()` to produce the report.
- Prints a confirmation message once complete.

This utility supports transparency and reproducibility in testing by making results and visuals easy to review
and share with team members or include in final project documentation.
"""


# tests/test_summary.py
import os
from datetime import datetime


def generate_summary_report(results: dict, plot_paths: list, output_path="tests/outputs/test_summary.md"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"#  Waste Management Test Summary\n")
        f.write(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("##  Test Results\n")
        for test_name, status in results.items():
            emoji = "✅" if status else "❌"
            f.write(f"- {emoji} {test_name}: {'Passed' if status else 'Failed'}\n")

        f.write("\n---\n")
        f.write("##  Visual Plots\n")
        for path in plot_paths:
            if os.path.exists(path):
                f.write(f"![{os.path.basename(path)}]({path})\n\n")

        f.write("\n---\n")
        f.write(" *Generated automatically by test_summary.py*\n")
# Add this to the bottom of tests/test_summary.py
if __name__ == "__main__":
    results = {
        "Clustering Test": True,
        "Route Optimization Test": True,
        "Fuel Cost Analysis": True
    }

    plots = [
        "tests/outputs/clustering_test_plot.png",
        "tests/outputs/route_optimization_test_plot.png"
    ]

    generate_summary_report(results, plots)
    print(" Summary report generated.")
