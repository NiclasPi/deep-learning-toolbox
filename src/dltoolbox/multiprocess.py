import psutil
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class MultiprocessConfig:
    """Configuration for efficient multiprocessing."""
    max_processes: int
    max_memory_per_process: int  # in bytes


def get_multiprocess_config(
        *,
        memory_utilization_limit: float = 0.8,
        max_available_memory: int | None = None,
        max_workers: int | None = None,
        physical_cores_only: bool = False,
) -> MultiprocessConfig:
    """
    Compute a multiprocessing configuration by distributing system resources
    (CPU cores and memory) evenly across worker processes.

    Args:
        memory_utilization_limit: Fraction of the system's available memory.
        max_available_memory: Absolute upper limit of memory usage in bytes.
        max_workers: Number of worker processes. Defaults to number of logical CPU cores.
        physical_cores_only: Only take physical cores into account.

    Returns:
        MultiprocessConfig:
            Configuration specifying the maximum number of processes (CPU cores)
            and the maximum memory available per process.
    """

    available_memory = int(psutil.virtual_memory().available * memory_utilization_limit)
    available_cpus = psutil.cpu_count(logical=not physical_cores_only)

    if max_available_memory is not None and max_available_memory < available_memory:
        available_memory = max_available_memory
    if max_workers is not None and max_workers < available_cpus:
        available_cpus = max_workers

    available_memory_per_process = available_memory // available_cpus

    return MultiprocessConfig(
        max_processes=available_cpus,
        max_memory_per_process=available_memory_per_process,
    )
