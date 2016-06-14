require 'DQN'



agent = DQNAgent({state_dim=3,n_actions=3})


s = torch.rand(3)

print(agent:act(s))

