from gym.envs.registration import register
from msevolution_env.envs.microstructure_evolution import MSEvolution, \
    MSEvolutionMultiGoal, MSEvolutionMultiGoalWGoalStates

register(
    id=MSEvolution.ENV_ID,
    entry_point='msevolution_env.envs:MSEvolution',
    max_episode_steps=2500,
)

register(
    id=MSEvolutionMultiGoal.ENV_ID,
    entry_point='msevolution_env.envs:MSEvolutionMultiGoal',
    max_episode_steps=2500,
)
register(
    id=MSEvolutionMultiGoalWGoalStates.ENV_ID,
    entry_point='msevolution_env.envs:MSEvolutionMultiGoalWGoalStates',
    max_episode_steps=2500,
)
