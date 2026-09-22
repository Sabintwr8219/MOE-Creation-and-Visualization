import time

import psutil


# Report elapsed time, throughput, CPU share, and memory use.
class Progress:
    def __init__(self, label, interval_seconds=5, include_children=False):
        self.label = label
        self.interval_seconds = interval_seconds
        self.include_children = include_children
        self.process = psutil.Process()
        self.started = time.perf_counter()
        self.last_report = self.started
        self.cpu_start = self.process.cpu_times()

    def report(self, completed, total=None, force=False):
        now = time.perf_counter()
        if not force and now - self.last_report < self.interval_seconds:
            return

        elapsed = max(now - self.started, 1e-9)
        cpu_now = self.process.cpu_times()
        cpu_seconds = cpu_now.user + cpu_now.system - self.cpu_start.user - self.cpu_start.system
        processes = [self.process]
        if self.include_children:
            processes.extend(self.process.children(recursive=True))
            for child in processes[1:]:
                try:
                    child_cpu = child.cpu_times()
                    cpu_seconds += child_cpu.user + child_cpu.system
                except psutil.Error:
                    pass
        cpu_share = 100 * cpu_seconds / elapsed / (psutil.cpu_count() or 1)
        ram_bytes = 0
        for process in processes:
            try:
                ram_bytes += process.memory_info().rss
            except psutil.Error:
                pass
        ram_gb = ram_bytes / 1024**3
        rate = completed / elapsed
        position = f"{completed:,}"
        if total:
            position += f"/{total:,} ({100 * completed / total:.1f}%)"

        print(
            f"{self.label}: {position} | {elapsed:.1f}s | "
            f"{rate:,.0f}/s | CPU: {cpu_share:.1f}% | RAM: {ram_gb:.2f} GB"
        )
        self.last_report = now

    @property
    def elapsed(self):
        return time.perf_counter() - self.started
