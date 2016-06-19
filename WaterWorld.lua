--[[
@Author: Ermano Arruda

An WaterWorld to play with the PuckWorld
]]

require 'torch'
require 'math'
require 'Environment'
require 'utils'

local WaterWorld = torch.class('WaterWorld','Environment')


function WaterWorld:__init(width,height)
    Environment.__init(self, width,height)

    self.max_elems = 20

    self.food_list = {}


    self.active_time = 50
    self.time = self.active_time
    self.inactive_list = {}

    self.z = torch.Tensor(self.agent.range_sensors:size(1)):fill(math.huge)
    self.fv = torch.Tensor(self.agent.range_sensors:size(1),2):fill(0)
    self.c = torch.Tensor(self.agent.range_sensors:size(1)*3):fill(0)
    self.sensed_ids = torch.LongTensor(self.agent.range_sensors:size(1)):fill(-1)
    self.sensed_types = torch.LongTensor(self.agent.range_sensors:size(1)):fill(-1)
    self.insersects = {}
    -- State dimenstions for each sensor (range,food_vx,food_vy,c,agent_vx,agent_vy)
    -- c = (wall,green,red), one-hot-encoding telling the type of sensed object
    self.state_dim = self.agent.range_sensors:size(1)*6 + 2
    self.state = torch.Tensor(self.state_dim)


    for i = 1, self.max_elems do
        local food = Food(false,self.world,love.physics.getMeter(),1.5)
        local poison = Food(true,self.world,love.physics.getMeter(),1.5)
        table.insert(self.food_list,food)
        table.insert(self.food_list,poison)

        food.fixture:setCategory(3)
        poison.fixture:setCategory(3)

        --print(poison.fixture:getCategory())
        food.fixture:setMask(poison.fixture:getCategory(),self.agent.fixture:getCategory())
        poison.fixture:setMask(poison.fixture:getCategory(),self.agent.fixture:getCategory())

        --Fixture:setMask( mask1, mask2, ... )
    end

end



function WaterWorld:getActions()
    return {{x = self.F,y = 0}, {x = -self.F, y = 0}, {x = 0, y = -self.F}, {x = 0, y = self.F}}
end

function WaterWorld:sampleAction()
    local actions = self:getActions()
    return torch.random( #actions )
end

function Environment:getStateDim()
    return self.state_dim
end

function Environment:getNActions()
    return #self:getActions()
end

function WaterWorld:getState()
    local measurements = self.z:clone()
    measurements:clamp(0,self.agent.range_limit)
    measurements = measurements:div(self.agent.range_limit)

    local velocities = self.fv:clone():div(self.ppm)
    local agent_vx, agent_vy = self.agent.body:getLinearVelocity()

    local one_hot_types = self:ToOneHot(self.sensed_types)
    --print(one_hot_types)
    self.state[1] = agent_vx/self.ppm
    self.state[2] = agent_vy/self.ppm

    local n = measurements:size(1)

    local idx = 3
    for i = 1, n do
        --print("idx: " .. idx .. " i: " .. i)
        self.state[idx] = measurements[i]
        self.state[idx+1] = one_hot_types[i][1]
        self.state[idx+2] = one_hot_types[i][2]
        self.state[idx+3] = one_hot_types[i][3]
        self.state[idx+4] = velocities[i][1]
        self.state[idx+5] = velocities[i][2]

        idx = idx + 6
    end

    return self.state
end

function WaterWorld:ToOneHot(types)
    local one_hot_types = torch.Tensor(types:size(1),3):fill(0)

    for i = 1, one_hot_types:size(1) do
        local c = types[i]
        
        if c > 0 then
            one_hot_types[i][c] = 1
        end

    end

    return one_hot_types

end


function WaterWorld:computeMeasurement(sensor_idx)

    local p = torch.Tensor({self.agent.body:getX(),self.agent.body:getY()})
    local sdir = self.agent.range_sensors:narrow(1,sensor_idx,1):view(-1)
    local food_idx = -1
    local z = self.agent.range_limit
    local best_intersect = nil
    local entity_type = 0 -- 0 - none, 1 - food_poison (green), 2 - food_good (red), 3 - wall


    for i = 1, self.max_elems do

        local food = self.food_list[i]

        if food.is_active then

            local food_p = torch.Tensor({food.body:getX(),food.body:getY()})
            local r = food.shape:getRadius()

            local intersect = utils.computeIntersect2(p[1],p[2],p[1]+sdir[1],p[2]+sdir[2],food_p[1],food_p[2],r)
            
            local intersect_left = utils.computeIntersectLine(p[1],p[2],sdir[1],sdir[2],0,0,1,0)
            local intersect_right = utils.computeIntersectLine(p[1],p[2],sdir[1],sdir[2],800,0,1,0)
            local intersect_top = utils.computeIntersectLine(p[1],p[2],sdir[1],sdir[2],0,0,0,1)
            local intersect_bottom = utils.computeIntersectLine(p[1],p[2],sdir[1],sdir[2],0,600,0,1)

            local intersects_walls = { intersect_left, intersect_right, intersect_top, intersect_bottom }

            if intersect then
                if intersect.t1 >= 0 and intersect.t1 < z then
                    z = intersect.t1
                    food_idx = i
                    best_intersect = intersect
                    entity_type = food.is_poison and 1 or 2
                end

                if intersect.t2 >= 0 and intersect.t2 < z then
                    z = intersect.t2
                    food_idx = i
                    best_intersect = intersect
                    entity_type = food.is_poison and 1 or 2
                end
            end

            -- checking walls intersect
            for j = 1, #intersects_walls do

                itsct_wall = intersects_walls[j]

                if itsct_wall.t3 >= 0 and itsct_wall.t3 < z then
                    z = itsct_wall.t3
                    food_idx = -1
                    best_intersect = itsct_wall
                    entity_type = 3 -- 2 for walls
                end

            end


        end

    end


    return z, food_idx, best_intersect, entity_type


end


function WaterWorld:computeMeasurements()

    local p = torch.Tensor({self.agent.body:getX(),self.agent.body:getY()})

    --print(self.agent.range_sensors:size())

    self.intersects = {}

    for i = 1, self.agent.range_sensors:size(1) do
        local sensor_dir = self.agent.range_sensors:narrow(1,i,1)

        local z, food_idx, intersect, entity_type = self:computeMeasurement(i)
        --print("entity type: " .. entity_type)
        self.z[i] = z  
        self.sensed_ids[i] = food_idx
        self.intersects[i] = intersect
        self.sensed_types[i] = entity_type

        if entity_type == 1 or entity_type == 2 then
            local vx, vy = self.food_list[food_idx].body:getLinearVelocity()
            self.fv[i][1] = vx
            self.fv[i][2] = vy
        else
            self.fv[i][1] = 0
            self.fv[i][2] = 0
        end

    end
end


function WaterWorld:computeRewardAndUpdate()

    self.time = self.time - 1

    local p = torch.Tensor({self.agent.body:getX(),self.agent.body:getY()})

    local reward = 0


    for i = 1, self.max_elems do


        local food = self.food_list[i]

        if food.is_active then

            local radius = self.agent.shape:getRadius() + food.shape:getRadius()

           
            food_p = torch.Tensor({food.body:getX(),food.body:getY()})

            local dir = food_p - p
            local d = dir:norm()
            
            if d < radius and food.is_poison then
                reward = reward - 1
                food.is_active = false
                table.insert(self.inactive_list, food)
                --print("colided with poison!")
            elseif d < radius and not food.is_poison then
                reward = reward + 1
                food.is_active = false
                table.insert(self.inactive_list, food)
               -- print("colided with food!")
            end
        end

    end


    if self.time <= 0 and #self.inactive_list > 0 then
        food = table.remove(self.inactive_list)
        food.is_active = true
        food:reinitRandom()
        self.time = self.active_time
    end

    self:computeMeasurements()


    return reward



end

function WaterWorld:drawAgentSensors()

    love.graphics.setColor(self.agent.range_sensor_color.r,self.agent.range_sensor_color.g,self.agent.range_sensor_color.b)
    local p = torch.Tensor({self.agent.body:getX(),self.agent.body:getY()})

    --print(self.agent.range_sensors:size())
    for i = 1, self.agent.range_sensors:size(1) do
        local sensor_dir = self.agent.range_sensors:narrow(1,i,1):view(-1)


        local z = self.z[i]
        local food_idx = self.sensed_ids[i]
        local sensed_type = self.sensed_types[i]

        if sensed_type > 0 and sensed_type < 3 and self.food_list[food_idx].is_active then -- if it's food and it's active

            local food = self.food_list[food_idx]

            love.graphics.setColor(food.color.r,food.color.g,food.color.b)
            local p2 = p + sensor_dir*z
            love.graphics.line(p[1], p[2], p2[1], p2[2])
        elseif sensed_type == 3 then -- if it's a wall

            love.graphics.setColor(100,100,100)
            local p2 = p + sensor_dir*z
            love.graphics.line(p[1], p[2], p2[1], p2[2])
        elseif sensed_type == 0 then -- if nothig is being sensed ...

            love.graphics.setColor(self.agent.range_sensor_color.r,self.agent.range_sensor_color.g,self.agent.range_sensor_color.b)
            local p2 = p + sensor_dir*self.agent.range_limit
            love.graphics.line(p[1], p[2], p2[1], p2[2])

        end

    end
    
end

function WaterWorld:draw()

    Environment.draw(self)

    self:drawAgentSensors()

    for i = 1, self.max_elems do
        if self.food_list[i].is_active then
            self.food_list[i]:draw()
        end
    end

    if self.intersects then 
        for i = 1, #self.intersects do
            local intersect = self.intersects[i]
            if intersect then
                love.graphics.circle( 'fill', intersect.x1, intersect.y1, 5, 10 )
                love.graphics.circle( 'fill', intersect.x2, intersect.y2, 5, 10 )
            end
        end
    end


end

function WaterWorld:step(action, dt)

    Environment.step(self,action,dt)
    
    reward = self:computeRewardAndUpdate()


    local reward = 0




    -- Compute reward


    return reward

end
