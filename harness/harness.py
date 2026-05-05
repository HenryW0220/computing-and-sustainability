from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
from zeus.monitor import ZeusMonitor

import json
import os
import subprocess
import sys
import time
import torch
import torch.nn as nn

class ProfilingPhase(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"


class Model(ABC):
    @abstractmethod
    def setup(self, env_name: str, **kwargs): ...

    @abstractmethod
    def train_step(self): ...

    @abstractmethod
    def eval(self, inp): ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class DebugModel(Model):
    def __init__(self, hdim=256, idim=84, odim=18):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn = nn.Sequential(
            nn.Linear(idim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, odim)
        ).to(self.device)
        self.opt = torch.optim.Adam(self.nn.parameters(), lr=1e-4)
        self.idim = idim
        self.odim = odim

    def setup(self, env_name, **kwargs):
        pass

    def train_step(self):
        BATCH_SIZE = 512
        inp = torch.randn(BATCH_SIZE, self.idim, device=self.device)
        target = torch.randint(0, self.odim, (BATCH_SIZE,), device=self.device)
        for _ in range(1000):
            loss = nn.CrossEntropyLoss()(self.nn(inp), target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        torch.cuda.synchronize()

    def eval(self, inp=None):
        if inp is None:
            inp = torch.randn(1, self.idim, device=self.device)
        with torch.no_grad():
            return self.nn(inp).argmax(dim=-1).item()

    @property
    def name(self):
        return "debug"

@contextmanager
def energy_window(monitor: ZeusMonitor, label: str, logfile: str):
    """Log the total joules over a profiling window"""

    # Start profiling
    monitor.begin_window(label)
    start_time = time.perf_counter()
    yield

    # Stop profiling
    duration_sec = time.perf_counter() - start_time
    measurement = monitor.end_window(label)
    record = {
        "label": label,
        "joules": measurement.total_energy,  # total joules used in window
        "seconds": duration_sec,             # seconds in measurement window
    }
    with open(logfile, "a") as file:
        json.dump(record, file)
        _ = file.write("\n")

def get_gpu_info():
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=name,power.limit,memory.total",
        "--format=csv,noheader,nounits"
    ]).decode().strip()
    return out

def get_env_info():
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version,
    }

def run(
    model: Model,
    env_name: str,
    num_train_steps: int,
    num_evals: int,
    outdir: str
):
    model.setup(env_name)
    monitor = ZeusMonitor(gpu_indices=[0], approx_instant_energy=True)
    logfile = os.path.join(outdir, "eco-profile.json")

    metadata = {
        "model": model.name,
        "env": env_name,
        "num_train_steps": num_train_steps,
        "gpu": get_gpu_info(),
    }
    with open(os.path.join(outdir, "meta.json"), "w") as file:
        json.dump(metadata, file)

    for step in range(num_train_steps):
        with energy_window(monitor, f"{ProfilingPhase.TRAINING}:step={step}", logfile):
            model.train_step()

    for step in range(num_evals):
        with energy_window(monitor, f"{ProfilingPhase.INFERENCE}:step={step}", logfile):
            model.eval(None)
