from pathlib import Path
from zeus.monitor import ZeusMonitor

from harness import energy_window, ProfilingPhase, DebugModel

import json


OUTPUT = Path("runs/debug_test")
OUTPUT.mkdir(parents=True, exist_ok=True)
LOG = str(OUTPUT / "energy.json")

model = DebugModel()
model.setup("fake-v0")
monitor = ZeusMonitor(gpu_indices=[0], approx_instant_energy=True)

for step in range(10):
    with energy_window(monitor, f"{ProfilingPhase.TRAINING}:step={step}", LOG):
        model.train_step()

for step in range(5):
    with energy_window(monitor, f"{ProfilingPhase.INFERENCE}:step={step}", LOG):
        model.eval(None)
