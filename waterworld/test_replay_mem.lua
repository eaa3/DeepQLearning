--[[
@Author: Ermano Arruda (exa371 at bham dot ac dot uk), June 2016

Testing replay memory
]]


require 'rl.ReplayMemory'
require 'rl.DQN'



function randomTransition(state_dim,n_actions)
	local transition = {}
	transition.s = torch.rand(state_dim)
	transition.a = torch.random(n_actions)
	transition.r = torch.rand(3)[1]
	transition.ns = torch.rand(state_dim)

	return transition
end

function test_sampling_one(mem)


	local ts = mem:sample_one(1)
	local ts2 = mem:sample_one(1)

	print("--TS1--")
	print("State: ",ts.s)
	print("Action: ", ts.a)
	print("Reward: ", ts.r)
	print("NextState: ",ts.ns)
	print("--TS2--")
	print("State: ",ts2.s)
	print("Action: ", ts2.a)
	print("Reward: ", ts2.r)
	print("NextState: ",ts2.ns)
end

function test_sampling_generic(mem,n_samples)

	local t = mem:sample(n_samples)

	print("State: ",t.s)
	print("Action: ", t.a)
	print("Reward: ", t.r)
	print("NextState: ", t.ns)


end

function test_write(mem)

	local file = torch.DiskFile("test_write.t7","w")

	mem:write(file)

	file:close()

end

function test_read(mem)

	local file = torch.DiskFile("test_write.t7","r")

	mem:read(file)

	file:close()


	print(mem.state_dim)
	print(mem.n_actions)
	print(mem.max_size)

end

opt = {state_dim=3,n_actions=3}
local mem = ReplayMemory(opt)

for i = 1, 10 do
	mem:insert(randomTransition(opt.state_dim,opt.n_actions))
end


--test_sampling_one(mem)
-- test_sampling_generic(mem,10)

-- test_sampling_generic(mem,1)


-- test_write(mem)

-- test_read(mem)


agent = DQNAgent(opt)

local t = mem:sample(1)

agent:computeGradient(t)






