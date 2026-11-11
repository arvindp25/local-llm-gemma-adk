import logging
from dataclasses import dataclass
from typing import Optional
from config.settings import get_settings
from core.exceptions import VRAMOverflowError

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

logger = logging.getLogger(__name__)


@dataclass
class GPUSnapshot:
    name: str
    total_mb: float
    used_mb: float
    free_mb: float
    utilization_gpu_pct: float
    temperature_c: float


class VRAMGuardian:
    """Monitors GPU memory usage and prevents OOM crashes on RTX 5050."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.settings = get_settings()
        self._initialized = False
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._initialized = True
            except Exception as e:
                logger.warning(f"Unable to initialize NVML for GPU monitor: {e}")

    def get_snapshot(self) -> Optional[GPUSnapshot]:
        if not self._initialized:
            return None
        try:
            name = pynvml.nvmlDeviceGetName(self._handle)
            name_str = name if isinstance(name, str) else name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            rates = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return GPUSnapshot(
                name=name_str,
                total_mb=round(mem.total / (1024 * 1024), 1),
                used_mb=round(mem.used / (1024 * 1024), 1),
                free_mb=round(mem.free / (1024 * 1024), 1),
                utilization_gpu_pct=float(rates.gpu),
                temperature_c=float(temp),
            )
        except Exception as e:
            logger.error(f"Error querying GPU snapshot: {e}")
            return None

    def assert_vram_safety(self, estimated_need_mb: float = 500.0) -> None:
        """Raises VRAMOverflowError if free VRAM is lower than safe threshold."""
        snapshot = self.get_snapshot()
        if snapshot is None:
            return

        free_headroom = snapshot.free_mb - estimated_need_mb
        min_required = self.settings.vram_headroom_mb

        if free_headroom < min_required:
            raise VRAMOverflowError(
                f"VRAM Safeguard Triggered! Available free VRAM: {snapshot.free_mb:.1f} MiB, "
                f"Required safety buffer: {min_required} MiB. Aborting turn to prevent CUDA OOM crash."
            )
