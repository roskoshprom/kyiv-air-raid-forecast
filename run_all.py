"""Reproduce everything from scratch with one command:  python run_all.py

download (pinned commit) -> inspection -> hourly grid -> leakage & timezone tests
-> EDA -> baselines + models on validation -> ONE final test evaluation.
Each step's output is saved to results/logs/.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "results" / "logs"

STEPS = [
    ("download", "src/download.py", False),
    ("step1_inspect", "src/step1_inspect.py", False),
    ("step2_build_grid", "src/step2_build_grid.py", False),
    ("test_leakage", "tests/test_leakage.py", False),
    ("test_timezone", "tests/test_timezone.py", False),
    ("step3_eda", "src/step3_eda.py", False),
    ("step4_validate", "src/step4_validate.py", False),
    ("step5_test", "src/step5_test.py", True),     # the only step allowed to open the test
]


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    for name, script, unlock in STEPS:
        env = {**os.environ, "PYTHONWARNINGS": "ignore"}
        env.pop("KSE_UNLOCK_TEST", None)
        if unlock:
            env["KSE_UNLOCK_TEST"] = "1"
        t0 = time.time()
        print(f"==> {name} ...", flush=True)
        r = subprocess.run([sys.executable, script], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        (LOGS / f"{name}.txt").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            sys.exit(f"step {name} failed (see results/logs/{name}.txt)")
        print(f"    ok in {time.time() - t0:.0f}s -> results/logs/{name}.txt")
    print("\nDONE. Key outputs: results/test_metrics.csv, results/validation_metrics.csv, figures/")


if __name__ == "__main__":
    main()
