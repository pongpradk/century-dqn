from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)

register(
    id="gymnasium_env/CenturyGolem-v0",
    entry_point="gymnasium_env.envs:CenturyGolemEnv",
)

register(
    id="gymnasium_env/CenturyGolem-v1",
    entry_point="gymnasium_env.envs:CenturyGolemEnvV1",
)

register(
    id="gymnasium_env/CenturyGolem-v2",
    entry_point="gymnasium_env.envs:CenturyGolemEnvV2",
)

register(
    id="gymnasium_env/CenturyGolem-v3",
    entry_point="gymnasium_env.envs:CenturyGolemEnv",
)

register(
    id="gymnasium_env/CenturyGolem-v4",
    entry_point="gymnasium_env.envs:CenturyGolemEnv",
)

register(
    id="gymnasium_env/CenturyGolem-v5",
    entry_point="gymnasium_env.envs:CenturyGolemEnv",
)

register(
    id="gymnasium_env/CenturyGolem-v6",
    entry_point="gymnasium_env.envs:CenturyGolemEnv",
)