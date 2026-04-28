"""
Test whether multiprocessing Pool works in parallel on this system.
Run: python test_parallel.py
Expected on working parallel: total ~2s, both workers have different PIDs
Expected if sequential:       total ~4s
"""
import time
import os
from multiprocessing import Pool, freeze_support, cpu_count


def worker(task_id):
    pid = os.getpid()
    start = time.time()
    time.sleep(2)
    elapsed = time.time() - start
    return task_id, pid, elapsed


if __name__ == '__main__':
    freeze_support()  # required for Windows frozen executables, harmless elsewhere

    print(f"Main PID: {os.getpid()}")
    print(f"CPU count: {cpu_count()}")
    print(f"Spawning Pool(2)...\n")

    t0 = time.time()
    with Pool(processes=2) as pool:
        results = pool.map(worker, [1, 2])
    total = time.time() - t0

    print(f"{'Task':<6} {'Worker PID':<12} {'Task time':<12}")
    print("-" * 32)
    for task_id, pid, elapsed in results:
        print(f"{task_id:<6} {pid:<12} {elapsed:.2f}s")

    print(f"\nTotal wall time: {total:.2f}s")
    print()

    pids = [r[1] for r in results]
    if total < 3.0:
        print("PARALLEL OK — both tasks ran simultaneously")
    else:
        print("SEQUENTIAL — tasks ran one after another")

    if len(set(pids)) > 1:
        print("DIFFERENT PIDs — separate worker processes confirmed")
    else:
        print("SAME PID — running in main process (no real parallelism)")
