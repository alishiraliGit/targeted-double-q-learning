import numpy as np

from rlcodebase.policies.base_policy import BasePolicy


class ArgMaxPolicy(BasePolicy):
    def __init__(self, critic):
        self.critic = critic

    def get_actions(self, ob_no: np.ndarray) -> np.ndarray:
        if ob_no.ndim < 2:
            ob_no = ob_no[np.newaxis, :]

        qa_values_na: np.ndarray = self.critic.qa_values(ob_no)

        ac_n = qa_values_na.argmax(axis=1)

        return ac_n

    def update(self, *args, **kwargs):
        pass
