from collections import defaultdict
from typing import Any
from typing import Callable
from typing import cast
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import warnings

import numpy as np

import optuna
from optuna._experimental import ExperimentalWarning
from optuna.distributions import BaseDistribution
import optuna._transform as transform
from optuna import distributions
from optuna.samplers._base import BaseSampler
from optuna.samplers._random import RandomSampler
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.study._multi_objective import _dominates
from optuna.trial import FrozenTrial
from optuna.trial import TrialState




class HarmonySearchSampler(BaseSampler):

    def __init__(
        self,
        *,
        HMCR: float = 0.5,
        PAR: float = 0.2,
        n_triggered_HMCR: int = 0,
        n_triggered_PAR: int = 0,
        band_width: float = 1e-2,
        harmony_memory_size: int = 2,
        max_iter_size: int = 6,
        harmonies_storage:  Optional[List[FrozenTrial]] = None,
        seed: Optional[int] = None,
        harmonies_pull: Optional[Dict[str, FrozenTrial]] = None,
        constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
    ) -> None:

        if not (HMCR is None or 0.0 <= HMCR <= 1.0):
            raise ValueError(
                "`HMCR` must be None or a float value within the range [0.0, 1.0]."
            )

        if not (PAR is None or 0.0 <= PAR <= 1.0):
            raise ValueError("`PAR` must be a float value within the range [0.0, 1.0].")


        if constraints_func is not None:
            warnings.warn(
                "The constraints_func option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

        self._HMCR = HMCR
        self._PAR = PAR
        self._n_triggered_HMCR = n_triggered_HMCR
        self._n_triggered_PAR = n_triggered_PAR
        self._band_width = band_width
        self._max_iter = max_iter_size
        self._random_sampler = RandomSampler(seed=seed)
        self._rng = np.random.RandomState(seed)
        self._constraints_func = constraints_func
        self.best_harmony = None
        self.worst_harmony = None
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)
        self._harmonies_pull = None
        self._harmony_memory_size = harmony_memory_size
        self._harmonies_storage = harmonies_storage

    def reseed_rng(self) -> None:
        self._random_sampler.reseed_rng()
        self._rng = np.random.RandomState()

    # def infer_relative_search_space(
    #     self, study: Study, trial: FrozenTrial
    # ) -> Dict[str, BaseDistribution]:
    #     return {}
    def infer_relative_search_space(self, study, trial):
        return optuna.samplers.intersection_search_space(study)

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space
    ) -> Dict[str, Any]:
        # trial_id = trial._trial_id
        completed_trials_num, random_trial = self._collect_previous_squad(study)
        # print('completed_trials_num:', completed_trials_num)
        params = {}

        if random_trial is None:
            for param_name, param_distribution in search_space.items():
                params[param_name].append(self._random_sampler.sample_independent(
                    study, trial, param_name, param_distribution
                ))
            return params
        
        random_trial_params = random_trial.params
        # print('random_trial_params', random_trial_params)
        # print('search space:', search_space.items())

        for param_name, param_distribution in search_space.items():
            # print('BEFORE COEF COMPUTE')
            self._HMCR = self.get_current_HMCR(completed_trials_num)
            self._PAR = self.get_current_PAR(completed_trials_num)
            # print('HMCR:', self._HMCR)
            # print('PAR:', self._PAR)
            hmcr_rnd = self._rng.uniform(0, 1)
            # print('hmcr_rnd', hmcr_rnd)
            if self._HMCR < hmcr_rnd:
                self._n_triggered_HMCR += 1
                # print('PASS HMCR')
                random_param = random_trial_params.get(param_name, None)
                if isinstance(param_distribution, distributions.CategoricalDistribution):
                    rnd_param_transformed = float(param_distribution.to_internal_repr(random_param))
                    rnd_for_test = self._rng.uniform(0, 1) 
                    self._n_triggered_HMCR += 1 if self._PAR <= rnd_for_test else 0
                    # print('PASS PAR:', rnd_for_test)
                    new_param_transformed = (rnd_param_transformed 
                                                # if self._PAR <= self._rng.uniform(0, 1) 
                                                if self._PAR <= rnd_for_test
                                                else (rnd_param_transformed + 
                                                        self._band_width*(self._rng.uniform(0, 1) - 0.5)
                                                        *(len(param_distribution.choices) - 1)))
                    new_param_transformed = ((len(param_distribution.choices) - 1) 
                                                if new_param_transformed > (len(param_distribution.choices) - 1) 
                                                else new_param_transformed)
                    params[param_name] = param_distribution.to_external_repr(new_param_transformed)
                else:     
                    rnd_param_transformed = transform._transform_numerical_param(random_param, param_distribution, transform_log=True)
    
                    new_param_transformed = (rnd_param_transformed 
                                                if self._PAR <= self._rng.uniform(0, 1) 
                                                else (rnd_param_transformed + 
                                                        self._band_width*(self._rng.uniform(0, 1) - 0.5)
                                                        *(param_distribution.high - param_distribution.low))) 
                    params[param_name] = transform._untransform_numerical_param(new_param_transformed, param_distribution, transform_log=True)
            else:
                params[param_name] = (self._random_sampler.sample_independent(
                    study, trial, param_name, param_distribution))
            
        return params

    def get_current_HMCR(self, completed_trials_num):
        hmcr = min(self._n_triggered_HMCR/self._harmony_memory_size, 1)
        if hmcr == 0:
            hmcr = 1/(1 + 1e-7 + np.exp(np.log(0.01*(self._max_iter-completed_trials_num))))
            # print('hmcr', hmcr)
        if hmcr == 1:
            hmcr = 1/(1 + 1e-7 + np.exp(-np.log(0.01*(self._max_iter-completed_trials_num))))
        return hmcr

    def get_current_PAR(self, completed_trials_num):
        par = min(self._n_triggered_PAR/self._harmony_memory_size, 1)
        if par == 0:
            par = 1/(1 + 1e-7 + np.exp(np.log(0.01*(self._max_iter-completed_trials_num))))
            # print('par', par)
        if par == 1:
            par = 1/(1 + 1e-7 + np.exp(-np.log(0.01*(self._max_iter-completed_trials_num))))
        return par


    def _collect_previous_squad(self, study: Study) -> Tuple[int, List[FrozenTrial]]:
        # randomly select one of previous trials
        trials = study.get_trials(deepcopy=False)

        completed_trials = []
        running_trials = []
        random_trial = None

        for trial in trials:

            if trial.state != optuna.trial.TrialState.COMPLETE:
                if trial.state == optuna.trial.TrialState.RUNNING:
                    running_trials.append(trial)
                continue

            completed_trials.append(trial)

        completed_trials.sort(key=lambda t: cast(float, t.values[0]), reverse=True)
        completed_trials_num = len(completed_trials)
        
        if completed_trials_num == 0:
            return completed_trials_num, random_trial

        if completed_trials_num == 1:
            return completed_trials_num, completed_trials[-1]
        
        if completed_trials_num <= self._harmony_memory_size:
            random_trial = completed_trials[(np.random.randint(completed_trials_num))]
        else:
            self._harmonies_storage = completed_trials[:self._harmony_memory_size]
            random_trial = self._harmonies_storage[(np.random.randint(self._harmony_memory_size))]

        return completed_trials_num, random_trial

    @staticmethod
    def _get_last_complete_trial(study):
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        return complete_trials[-1]

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )