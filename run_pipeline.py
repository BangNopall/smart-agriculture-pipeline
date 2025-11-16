import subprocess
import sys

scripts = [
    "scripts/extract_bps.py",
    "scripts/extract_bmkg.py",
    "scripts/transform.py",
    "scripts/load_db.py",
    "scripts/model_training.py"
]

print("="*60)
print("SMART AGRICULTURE PIPELINE")
print("="*60)

for script in scripts:
    print(f"\n▶ Running: {script}")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ {script} completed successfully")
        print(result.stdout)
    else:
        print(f"✗ {script} failed")
        print(result.stderr)
        break

print("\n" + "="*60)
print("PIPELINE COMPLETED!")
print("="*60)