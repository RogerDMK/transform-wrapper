from core.environments.env_obj import DiscretePointEnv
from core.wrappers.transform_matrix import transformation_wrapper

env = DiscretePointEnv(ndim=4)
env = transformation_wrapper(env)

obs, info = env.reset()

obs, _, _, _, info = env.step(1,1)

print(obs)
print(info)