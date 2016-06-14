require 'torch'
require 'nn'
require 'ReplayMemory'

local DQNAgent = torch.class('DQNAgent')


function DQNAgent:__init(opt)

    self.state_dim = opt.state_dim
    self.n_actions = opt.n_actions

    self.gamma = opt.gamma or 0.95
    self.eta = opt.eta or 0.001

    self.episilon = opt.episilon or 0.2
    self.eps_start = opt.eps_start or 0.2
    self.eps_end = opt.eps_end or 0.2--0.05
    self.eps_end_t = opt.eps_end_t or 100000

    self.n_replay = opt.n_replay or 1
    self.batch_size = opt.batch_size or 20
    self.n_hidden_units = opt.n_hidden_units or 120

    self.swap_target_qnet_every = 1

    self.experience_add_every = 10
    self.experience_count_down = self.experience_add_every
    self.learn_every = 10

    self.max_delta = opt.max_delta or 1

    self.n_steps = 1


    self.lr_start       = opt.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = opt.lr_end or 0.001
    self.lr_endt        = opt.lr_endt or 10000000
    self.wc             = opt.wc or 0.000001  -- L2 weight cost.


    self.qnet = self:createNet()
    self.target_qnet = self.qnet:clone()


    self.replay_mem = ReplayMemory({state_dim=self.state_dim,n_actions=self.n_actions})

    self.w, self.dw = self.qnet:getParameters()
    self.dw:zero()

    self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    self.current_tderr = 0

    self.average_reward = -3

    self.s = torch.Tensor()
    self.a = nil
    self.r = nil
    self.nr = nil
    self.ns = torch.Tensor()
    self.na = nil

    self.recent_qs = torch.Tensor()




end


function DQNAgent:createNet()
    local net = nn.Sequential()
    net:add(nn.Reshape(self.state_dim))
    net:add(nn.Linear(self.state_dim, self.n_hidden_units))
    --net:add(nn.ReLU(true))
    net:add(nn.Tanh())
    -- net:add(nn.Sigmoid())
    --net:add(nn.Linear(self.n_hidden_units, self.n_hidden_units))
    -- net:add(nn.Tanh())
    --net:add(nn.ReLU(true))
    net:add(nn.Linear(self.n_hidden_units, self.n_actions))
    --net:add(nn.SoftMax())
-- net:add(nn.ReLU(true))
--     net:add(nn.Linear(self.n_hidden_units, self.n_hidden_units))
--     net:add(nn.ReLU(true))
    return net
end

function DQNAgent:setInitialState(s)
    self.s = torch.Tensor({s})
end

function DQNAgent:select_greedy(qs)
    
    _, ind = torch.max(qs,1)

    return ind[1]
end

function DQNAgent:select_egreedy(qs)
    r = torch.uniform()

    self.episilon = (self.eps_end +
                math.max(0, (self.eps_start - self.eps_end) * (self.eps_end_t -
                math.max(0, self.n_steps)) / self.eps_end_t))

    if r < self.episilon then
        return torch.random(self.n_actions)
    else
        return self:select_greedy(qs)
    end

end

function DQNAgent:select_prob(qs)
    r = torch.uniform()
    acc = 0
    i = 1
    while r > acc and i < qs:size(1) do
        acc = acc + qs[i]
        i = i+1
    end

    return i
end


function DQNAgent:act(s, greedy)

    local qs = self.qnet:forward(s)

    local a
    if greedy then
        a = self:select_egreedy(qs)
    else
        a = self:select_egreedy(qs)
    end

    self.transition_buf = {}
    self.transition_buf.s = s:clone()
    self.transition_buf.a = torch.Tensor({a})[1]

    self.recent_qs = qs:clone()

    self.n_steps = self.n_steps + 1

    -- state switch
    self.s = self.ns:clone()
    self.ns = s:clone()
    self.a = self.na
    self.na = torch.Tensor({a})[1]

    return a

end



function DQNAgent:perceive(obs)
    local nr = obs.r

    if self.r ~= nil then

        if self.average_reward == nil then
            self.average_reward = r
        else
            self.average_reward = self.average_reward * 0.999 + r * 0.001
        end
    

        self.transition_buf = {}
        self.transition_buf.s = self.s:clone()
        self.transition_buf.a = self.a
        self.transition_buf.r = self.r
        self.transition_buf.ns = self.ns:clone()

        local tderror
        if self.eta > 0 then
            local trans_batch = {s = torch.zeros(1,self.state_dim):copy(self.s), a = torch.Tensor({self.a}), r = torch.Tensor({self.r}), ns = torch.zeros(1,self.state_dim):copy(self.ns)}
            tderror = self:learnFromTransition(trans_batch)
            --print("tderr: " .. tderror)
            if self.replay_mem.n_entries > 2 then
                trans_batch = self.replay_mem:sample(1,true)--self.replay_mem:getRecent(1)
                self:learnFromTransition(trans_batch)
            end
        end
        --  and math.abs(tderror) >= 0.00001 
        if self.experience_count_down == 0 then
            self.replay_mem:insert(self.transition_buf)
            self.experience_count_down = self.experience_add_every
        else
            self.experience_count_down = self.experience_count_down - 1
        end

        if self.eta > 0 then
            agent:qLearnMiniBatch()
        end
    end

    self.r = nr



end

function DQNAgent:computeUpdate(trans_batch)

    local r, s, a, ns

    r = trans_batch.r
    s = trans_batch.s
    a = trans_batch.a
    ns = trans_batch.ns

    -- delta = r + gamma* max_a' q_target(s',a') - q(s,a)
    delta = r:clone()
    
    local target_qs = self.target_qnet:forward(ns)
    local max_q = target_qs:max(2)
    
    local qs = self.qnet:forward(s)

    local q = torch.Tensor(qs:size(1))

    for i=1, qs:size(1) do
        q[i] = qs[i][a[i]]
    end

    delta:add(self.gamma,max_q)

    -- delta += -q
    delta:add(-1,q)

    if self.clip_delta then
       delta[delta:ge(self.clip_delta)] = self.clip_delta
       delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local batch_size = trans_batch.a:size(1)
    local targets = torch.zeros(batch_size, self.n_actions)
    for i=1, batch_size do
        targets[i][a[i]] = delta[i] -- I thought we should use qs, instead of those targets, why?
    end


    return delta, targets



end


function DQNAgent:learnFromTransition(transition)

    local delta, targets = self:computeUpdate(transition)
    
    local mean_tderror = delta:mean()

    self:RMSPropUpdate(transition,targets)
    --self:SimpleGradientUpdate(transition,targets)

    return mean_tderror

end

function DQNAgent:SimpleGradientUpdate(trans_batch,targets)

    self.dw:zero()
    self.qnet:backward(trans_batch.s,targets)
    
    self.dw:add(-0.01, self.w) -- Weight decay
    self.w:add(-self.eta, self.dw)
end

function DQNAgent:RMSPropUpdate(trans_batch,targets)

    self.dw:zero()
    self.qnet:backward(trans_batch.s,targets)

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.n_steps)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                        self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- Adam, adaptation of Hinton's RMSprop (Geoff Hinton)
    -- use gradients
    self.g:mul(0.95):add(0.05, self.dw)
    self.tmp:cmul(self.dw, self.dw)
    self.g2:mul(0.95):add(0.05, self.tmp)
    self.tmp:cmul(self.g, self.g)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(0.001) -- 0.01
    self.tmp:sqrt()

    -- accumulate update
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
    self.w:add(self.deltas)

end

function DQNAgent:qLearnMiniBatch()

    if self.replay_mem.n_entries >= self.batch_size and self.n_steps%self.learn_every == 0 then

        for i = 1, self.n_replay do
            trans_batch = self.replay_mem:sample(self.batch_size,false)

            local delta, targets = self:computeUpdate(trans_batch)

            self.tderr_avg = delta:clone():abs():mean()
            self.current_tderr = delta:mean()

            self:RMSPropUpdate(trans_batch,targets)
            --self:SimpleGradientUpdate(trans_batch,targets)

        end

        if self.n_steps%self.swap_target_qnet_every == 0 then
            self.target_qnet = self.qnet:clone('weight','bias')
            --self.w, self.dw = self.qnet:getParameters()
        end
        


    end
end

function DQNAgent:save(filename)
    torch.save(filename, self.qnet)
end

function DQNAgent:load(filename)
    self.qnet = torch.load(filename)
    self.target_qnet = self.qnet:clone('weight','bias')
end



