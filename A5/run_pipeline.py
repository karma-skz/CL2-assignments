import subprocess
scripts = ["parse_to_rs3.py", "extract_rst_info.py", "rst_summarizer.py", "metrics.py"]

for script in scripts:
    subprocess.run(["python3", script])
print(f"completed")