from optuna import distributions
from optuna.samplers import BaseSampler
from optuna.study import StudyDirection
from optuna.trial import TrialState
import optuna
import numpy as np


class SimulatedAnnealingSampler(BaseSampler):
    def __init__(self, temperature=100, cooldown_factor=0.9, neighbor_range_factor=0.1, seed=None):
        self._rng = np.random.RandomState(seed)
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)
        self._temperature = temperature
        self.cooldown_factor = cooldown_factor
        self.neighbor_range_factor = neighbor_range_factor
        self._current_trial = None

    def infer_relative_search_space(self, study, trial):
        return optuna.samplers.intersection_search_space(study)

    def sample_relative(self, study, trial, search_space):
        if search_space == {}:
            return {}

        prev_trial = self._get_last_complete_trial(study)

        if self._rng.uniform(0, 1) <= self._transition_probability(study, prev_trial):
            self._current_trial = prev_trial

        params = self._sample_neighbor_params(search_space)

        self._temperature *= self.cooldown_factor

        return params

    def _sample_neighbor_params(self, search_space):
        params = {}
        for param_name, param_distribution in search_space.items():
            if isinstance(param_distribution, distributions.CategoricalDistribution):
                neighbor_low = neighbor_high = None
            else:               
                current_value = self._current_trial.params[param_name]
                width = (
                    param_distribution.high - param_distribution.low
                ) * self.neighbor_range_factor
                neighbor_low = max(current_value - width, param_distribution.low)
                neighbor_high = min(current_value + width, param_distribution.high)
                params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)

        return params

    def _transition_probability(self, study, prev_trial):
        if self._current_trial is None:
            return 1.0

        prev_value = prev_trial.value
        current_value = self._current_trial.value

        if study.direction == StudyDirection.MINIMIZE and prev_value <= current_value:
            return 1.0
        elif study.direction == StudyDirection.MAXIMIZE and prev_value >= current_value:
            return 1.0

        return np.exp(-abs(current_value - prev_value) / self._temperature)

    @staticmethod
    def _get_last_complete_trial(study):
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        return complete_trials[-1]

    def sample_independent(self, study, trial, param_name, param_distribution):

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )