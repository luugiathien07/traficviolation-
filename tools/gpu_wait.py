"""Wait until no other process holds a large share of GPU memory (another training run).
Two trainings on one 8 GB GPU run out of memory or crash with CUDA errors."""
import subprocess
import time


def wait_for_free_gpu(max_other_mib: int = 1500, poll_s: int = 60) -> None:
    announced = False
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.TimeoutExpired):
            return
        busy = [line for line in out.splitlines() if line.strip() and int(line.split(",")[1]) > max_other_mib]
        if not busy:
            return
        if not announced:
            print(f"GPU đang bận ({len(busy)} tiến trình dùng > {max_other_mib} MiB); chờ...", flush=True)
            announced = True
        time.sleep(poll_s)
