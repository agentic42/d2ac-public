def get_observation_dim(observation_space):
    return observation_space.shape[1]


def assert_uniform_action_space(action_space):
    act_dim = action_space.shape[1]
    act_limit = action_space.high[0][0]
    assert action_space.high.ndim == action_space.low.ndim == 2
    assert all(i == act_limit for i in action_space.high[0])
    assert all(i == -act_limit for i in action_space.low[0])
    return act_dim, act_limit
