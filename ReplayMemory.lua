--[[
@Author: Ermano Arruda

An implementation of ReplayMemory for the DQN algorithm proposed by Mnih et al
]]



require 'torch'


local ReplayMemory = torch.class("ReplayMemory")


function ReplayMemory:__init(args)


	self.max_size = args.max_size or 5000

	self.recent_mem_size = args.recent_mem_size or 200

	self.state_dim = args.state_dim
	self.n_actions = args.n_actions

	self.n_entries = 0
	self.next_index = 1


	self.s = torch.Tensor(self.max_size, self.state_dim):fill(0)
	self.a = torch.LongTensor(self.max_size):fill(0)
	self.r = torch.zeros(self.max_size)
	self.ns = torch.Tensor(self.max_size, self.state_dim):fill(0)

end

function ReplayMemory:insert(transition)

	self.s:narrow(1, self.next_index, 1):copy(transition.s)
	self.a[self.next_index] = transition.a
	self.r[self.next_index] = transition.r
	self.ns:narrow(1, self.next_index, 1):copy(transition.ns)

	self.next_index = (self.next_index+1)%self.max_size
	if self.next_index == 0 then
		self.next_index = 1
	end

	self.n_entries = self.n_entries + 1


end

function ReplayMemory:getRecent(size)

	assert(self.n_entries >= size)

	local trans_batch = {}

	trans_batch.s = torch.zeros(size,self.state_dim)
	trans_batch.a = torch.LongTensor(size):fill(0)
	trans_batch.r = torch.zeros(size)
	trans_batch.ns = torch.zeros(size,self.state_dim)

	base = math.max(self.next_index - size,1)
	limit = base + size - 1
	i = 1
	for idx = base, limit do
		trans = self:getTransition(idx)
		--print("got " .. idx)

		trans_batch.s[i]:copy(trans.s)
		trans_batch.a[i] = trans.a
		trans_batch.r[i] = trans.r
		trans_batch.ns[i]:copy(trans.ns)

		i = i + 1
	end

	return trans_batch


end

function ReplayMemory:getTransition(idx)
	local transition = {}

	transition.s = self.s:narrow(1, idx, 1)
	transition.a = self.a[idx]
	transition.r = self.r[idx]
	transition.ns = self.ns:narrow(1, idx, 1)

	return transition
end

function ReplayMemory:sample_one(recent)

	assert(self.n_entries > 0)

	local base = math.max(self.n_entries-self.recent_mem_size,0)

	local idx
	if recent then
		idx = torch.random(base, self.n_entries)%self.max_size
	else
		idx = torch.random(1,self.n_entries)%self.max_size
	end

	if idx == 0 then
		idx = 1
	end

	return self:getTransition(idx)
end

function ReplayMemory:sample(size,recent)

	assert(self.n_entries >= size)

	local trans_batch = {}

	trans_batch.s = torch.zeros(size,self.state_dim)
	trans_batch.a = torch.LongTensor(size):fill(0)
	trans_batch.r = torch.zeros(size)
	trans_batch.ns = torch.zeros(size,self.state_dim)

	for i = 1, size do
		trans = self:sample_one(recent)

		trans_batch.s[i]:copy(trans.s)
		trans_batch.a[i] = trans.a
		trans_batch.r[i] = trans.r
		trans_batch.ns[i]:copy(trans.ns)
	end

	return trans_batch
end

function ReplayMemory:write(file)
	file:writeObject({self.state_dim,
                      self.n_actions,
                      self.max_size})
end

function ReplayMemory:read(file)
	local state_dim, n_actions, max_size = unpack(file:readObject())

	self.state_dim = state_dim
	self.n_actions = n_actions
	self.max_size = max_size

	self.n_entries = 0
	self.next_index = 0

	self.s = torch.Tensor(self.max_size, self.state_dim):fill(0)
	self.a = torch.LongTensor(self.max_size):fill(0)
	self.r = torch.zeros(self.max_size)
	self.ns = torch.Tensor(self.max_size, self.state_dim):fill(0)
end

