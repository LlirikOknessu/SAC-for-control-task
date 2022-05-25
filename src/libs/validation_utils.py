import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ResponseDict:
    time_ms: list = field(default_factory=list)
    action: list = field(default_factory=list)
    current: list = field(default_factory=list)
    position: list = field(default_factory=list)
    angular_velocity: list = field(default_factory=list)
    object_velocity: list = field(default_factory=list)

    def from_dict(self, response_dict: dict):
        self.time_ms = response_dict.get('time_ms')
        self.action = response_dict.get('action')
        self.current = response_dict.get('current')
        self.position = response_dict.get('position')
        self.angular_velocity = response_dict.get('angular_velocity')
        self.object_velocity = response_dict.get('object_velocity')
        return self


def save_step_response(file_path: Path, response_dict: ResponseDict):
    df = pd.DataFrame([])
    actions_array = np.array(response_dict.action).squeeze()
    df = df.assign(time=response_dict.time_ms)
    df = df.assign(action=actions_array)
    df = df.assign(current=response_dict.current)
    df = df.assign(position=response_dict.position)
    df = df.assign(angular_velocity=response_dict.angular_velocity)
    df = df.assign(object_velocity=response_dict.object_velocity)
    df.to_csv(file_path, index=False)
