--[[
@Author: Ermano Arruda (exa371 at bham dot ac dot uk), June 2016

Testing the agent
]]

require 'rl.DQN'



agent = DQNAgent({state_dim=3,n_actions=3})


s = torch.rand(3)

print(agent:act(s))

