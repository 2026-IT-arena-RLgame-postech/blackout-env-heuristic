import numpy as np

from blackout_env import BaseModel


class MyPolicy(BaseModel):
    def __init__(self) -> None:
        super(BaseModel, self).__init__()

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        pass
