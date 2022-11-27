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
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_GENERATION_KEY = "pso:generation"


class ParticleSwarmSampler(BaseSampler):

    def __init__(
        self,
        *,
        particles_num: int = 5,
        inertia_w: float = 0.5,
        cognitive_coef: float = 0.2,
        social_coef: float = 0.3,
        speed_max: float = 0.9,
        seed: Optional[int] = None,
        constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
        particles_velocities: Optional[Dict[str, BaseDistribution]] = None
    ) -> None:

        if not isinstance(particles_num, int):
            raise TypeError("`particles_num` must be an integer value.")

        if particles_num < 2:
            raise ValueError("`particles_num` must be greater than or equal to 2.")

        if not (inertia_w is None or 0.0 <= inertia_w <= 1.0):
            raise ValueError(
                "`inertia_w` must be None or a float value within the range [0.0, 1.0]."
            )

        if not (0.0 <= cognitive_coef <= 2.0):
            raise ValueError("`cognitive_coef` must be a float value within the range [0.0, 2.0].")

        if not (0.0 <= social_coef <= 2.0):
            raise ValueError("`social_coef` must be a float value within the range [0.0, 2.0].")

        if constraints_func is not None:
            warnings.warn(
                "The constraints_func option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

        self._particles_num = particles_num
        self._inertia_w = inertia_w
        self._cognitive_coef = cognitive_coef
        self._social_coef = social_coef
        self._speed_max = speed_max
        self._random_sampler = RandomSampler(seed=seed)
        self._rng = np.random.RandomState(seed)
        self._constraints_func = constraints_func
        self.particles_velocities: Dict[str, Dict[str: float]]
        self.best_in_generation = None
        self.best_in_population = None
        self.generation_num = -1
        self.particles_velocities = {}
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)

    def reseed_rng(self) -> None:
        self._random_sampler.reseed_rng()
        self._rng = np.random.RandomState()

    def infer_relative_search_space(self, study, trial):
        return optuna.samplers.intersection_search_space(study)

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        trial_id = trial._trial_id
        previous_generation = self._collect_previous_squad(study)
        params = {}

        if self.best_in_population is None or previous_generation is None:
            generation = self.generation_num = 0
            study._storage.set_trial_system_attr(trial_id, _GENERATION_KEY, generation)
            for param_name, param_distribution in search_space.items():
                params.update({param_name: self._random_sampler.sample_independent(
                    study, trial, param_name, param_distribution
                    )})
                init_random_velocity = np.random.ranf(1)[0]
                if trial_id in self.particles_velocities:
                    self.particles_velocities[trial_id].update({param_name: 
                                                        (init_random_velocity 
                                                        if init_random_velocity <= self._speed_max 
                                                        else self._speed_max)})
                else:
                    self.particles_velocities[trial_id] = {param_name: 
                                                        (init_random_velocity 
                                                        if init_random_velocity <= self._speed_max 
                                                        else self._speed_max)}
            return params
        
        
        generation = self.generation_num + 1
        study._storage.set_trial_system_attr(trial_id, _GENERATION_KEY, generation)

        previous_particle_params = previous_generation.params
        pbest_params = self.best_in_population.params
        gbest_params = self.best_in_population.params
        previous_particle_velocities = self.particles_velocities.get(previous_generation._trial_id)

        for param_name, param_distribution in search_space.items():
            previous_param = previous_particle_params.get(param_name, None)
            pbest_param = pbest_params.get(param_name, None)
            gbest_param = gbest_params.get(param_name, None)
            previous_param_velocity = previous_particle_velocities.get(param_name, None)

            if isinstance(param_distribution, distributions.CategoricalDistribution):
                previous_param_transformed = float(param_distribution.to_internal_repr(previous_param))
                pbest_param_transformed = float(param_distribution.to_internal_repr(pbest_param))
                gbest_param_transformed = float(param_distribution.to_internal_repr(gbest_param))
                new_velocity = (self._inertia_w * previous_param_velocity + 
                                self._cognitive_coef*np.random.ranf(1)[0]*(pbest_param_transformed - previous_param_transformed) + 
                                self._social_coef*np.random.ranf(1)[0]*(gbest_param_transformed-previous_param_transformed))
                new_param_transformed = previous_param_transformed + new_velocity
                new_param_transformed = ((len(param_distribution.choices) - 1) 
                                            if new_param_transformed > (len(param_distribution.choices) - 1) 
                                            else new_param_transformed)
                if trial_id in self.particles_velocities:
                    self.particles_velocities[trial_id].update({param_name: new_velocity})
                else:
                    self.particles_velocities[trial_id] = {param_name: new_velocity}
                params[param_name] = param_distribution.to_external_repr(new_param_transformed)
            else:     
                previous_param_transformed = transform._transform_numerical_param(previous_param, param_distribution, transform_log=True)
                pbest_param_transformed = transform._transform_numerical_param(pbest_param, param_distribution, transform_log=True)
                gbest_param_transformed = transform._transform_numerical_param(gbest_param, param_distribution, transform_log=True)
                new_velocity = (self._inertia_w * previous_param_velocity + 
                                self._cognitive_coef*np.random.ranf(1)[0]*(pbest_param_transformed - previous_param_transformed) + 
                                self._social_coef*np.random.ranf(1)[0]*(gbest_param_transformed-previous_param_transformed))
                new_param_transformed = previous_param_transformed + new_velocity
                if trial_id in self.particles_velocities:
                    self.particles_velocities[trial_id].update({param_name: new_velocity})
                else:
                    self.particles_velocities[trial_id] = {param_name: new_velocity}                
                params[param_name] = transform._untransform_numerical_param(new_param_transformed, param_distribution, transform_log=True)
                

        return params


    def _collect_previous_squad(self, study: Study) -> List[FrozenTrial]:
        trials = study.get_trials(deepcopy=False)

        generation_to_runnings = defaultdict(list)
        generation_to_particles = defaultdict(list)
        generation = -1
        for trial in trials:
            if _GENERATION_KEY not in trial.system_attrs:
                continue

            generation = trial.system_attrs[_GENERATION_KEY]
            if trial.state != optuna.trial.TrialState.COMPLETE:
                if trial.state == optuna.trial.TrialState.RUNNING:
                    generation_to_runnings[generation].append(trial)
                continue

            generation_to_particles[generation].append(trial)

        previous_generation: List[FrozenTrial] = []
        while True:
            particles = generation_to_particles[generation] 
            previous_generation = particles[-1] if particles else None

            if len(particles) < self._particles_num:
                break
            
            self.generation_num = self.generation_num + 1
            self.best_in_generation = study.best_trial
            self.best_in_population = study.best_trial_from_last_n(last_n=self._particles_num)
            break

        return previous_generation

    @staticmethod
    def _get_last_complete_trial(study):
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        return complete_trials[-1]

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )