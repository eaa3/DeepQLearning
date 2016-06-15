require 'torch'


local CircularBuffer = torch.class('CircularBuffer')


function CircularBuffer:__init(max_size)
	self.buffer = torch.Tensor(max_size)
	self.read_idx = 0
	self.write_idx = 0

	self.n_elems = 0

end

function CircularBuffer:insert(value)

	self.buffer[self.write_idx+1] = value

	self.write_idx = (self.write_idx+1)%self.buffer:size(1)

	self.n_elems = math.min(self.n_elems + 1,self.buffer:size(1))

end

function CircularBuffer:queryAll()

	local out = torch.Tensor(math.min(self.buffer:size(1),self.n_elems))

	local i = 0
	if self.n_elems == self.buffer:size(1) then
		i = self.write_idx
	end
	
	local k = 1
	for k=1,out:size(1) do

		out[k] = self.buffer[i+1]

		i = (i+1)%self.buffer:size(1)
	end

	return out
end

function CircularBuffer:get(idx)
	local base = self:getStartIdx()

	local i = (base+idx-1)%self.buffer:size(1) + 1

	return self.buffer[i]
end

function CircularBuffer:maxSize()
	return self.buffer:size(1)
end

function CircularBuffer:count()
	return self.n_elems
end

function CircularBuffer:getStartIdx()
	local i = 0
	if self.n_elems == self.buffer:size(1) then
		i = self.write_idx
	end

	return i
end
