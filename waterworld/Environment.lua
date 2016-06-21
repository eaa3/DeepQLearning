--[[
@Author: Ermano Arruda

An WaterWorld to play with the PuckWorld
]]

require 'torch'
require 'math'
require 'Food'


local Environment = torch.class("Environment")



local PPM = 64 -- Pixel per meter ratio
local F = 8*PPM


function Environment:__init(width,height)

	self.F = F
	self.ppm = PPM

    self.boundaries = {}
    self.agent = {}
    local gx = 0.0
    local gy = 0.0 -- 9.81*64 (9.81 m/s*s)
    self.world = love.physics.newWorld(0, 0, true) 
    love.physics.setMeter(PPM) -- 1 meter in the world is equals to 64px in the scre
    self.boundaries = {{},{},{},{}}
    -- top
    self.boundaries[1].shape = love.physics.newEdgeShape( 0, 0, width, 0 )
    self.boundaries[1].body = love.physics.newBody( self.world, 0, 0, "static" )
    self.boundaries[1].fixture = love.physics.newFixture( self.boundaries[1].body, self.boundaries[1].shape, 1 )
   	self.boundaries[1].fixture:setCategory(2)
   	self.boundaries[1].fixture:setFriction(0)
    self.boundaries[1].fixture:setRestitution(1.0)
    -- right
    self.boundaries[2].shape = love.physics.newEdgeShape( width, 0, width, height )
    self.boundaries[2].body = love.physics.newBody( self.world, 0, 0, "static" )
    self.boundaries[2].fixture = love.physics.newFixture( self.boundaries[2].body, self.boundaries[2].shape, 1 )
    self.boundaries[2].fixture:setCategory(2)
    self.boundaries[2].fixture:setFriction(0)
    self.boundaries[2].fixture:setRestitution(1.0)
    -- bottom
    self.boundaries[3].shape = love.physics.newEdgeShape( 0, height, width, height )
    self.boundaries[3].body = love.physics.newBody( self.world, 0, 0, "static" )
    self.boundaries[3].fixture = love.physics.newFixture( self.boundaries[3].body, self.boundaries[3].shape, 1)
    self.boundaries[3].fixture:setCategory(2)
    self.boundaries[3].fixture:setFriction(0)
    self.boundaries[3].fixture:setRestitution(1.0)
    -- left
    self.boundaries[4].shape = love.physics.newEdgeShape( 0, 0, 0, height )
    self.boundaries[4].body = love.physics.newBody( self.world, 0, 0, "static" )
    self.boundaries[4].fixture = love.physics.newFixture( self.boundaries[4].body, self.boundaries[4].shape, 1)
    self.boundaries[4].fixture:setCategory(2)
    self.boundaries[4].fixture:setFriction(0)
    self.boundaries[4].fixture:setRestitution(1.0)
    
    -- agent
    self.agent.shape = love.physics.newCircleShape( 15 )
    self.agent.body = love.physics.newBody( self.world, 100, 0, "dynamic" )
    self.agent.body:setLinearDamping(1.5)
    self.agent.body:setMass(10)
    -- self.agent.fixture:setFriction( .9 )
    self.agent.fixture = love.physics.newFixture( self.agent.body, self.agent.shape, 1 )
    self.agent.fixture:setRestitution(0.05)
    self.agent.fixture:setCategory(1)
    self.agent.color = {r=14, g=47, b=193}

    self.agent.n_sensors = 30
    local range_x = torch.cos(torch.linspace(0,2*math.pi,self.agent.n_sensors))
    local range_y = -torch.sin(torch.linspace(0,2*math.pi,self.agent.n_sensors))
    self.agent.range_sensors = torch.cat(range_x,range_y,2)
    self.agent.range_sensor_color = {r=200, g=200, b=200}
    self.agent.range_limit = 125


    function self.agent:draw()
	    love.graphics.setColor(self.color.r, self.color.g, self.color.b) 
	    love.graphics.circle("fill", self.body:getX(), self.body:getY(), self.shape:getRadius())
	end

	function self.agent:applyForce(fx,fy)
     	self.body:applyForce(fx,fy)
	end


end


function Environment:getActions()

	return {{x = F,y = 0}, {x = -F, y = 0}, {x = 0, y = -F}, {x = 0, y = F}, {x = 0, y = 0.0}}
end

function Environment:sampleAction()
	local actions = self:getActions()
	return torch.random( #actions )
end

function Environment:getStateDim()
    return #self:getState()
end

function Environment:getNActions()
    return #self:getActions()
end

function Environment:getState()
    return {}
end

function Environment:draw()

	self.agent:draw()
end

function Environment:step(action, dt)

	local reward = 0

    -- Agent performs action on world
    local force = self:getActions()[action]
    self.agent:applyForce(force.x,force.y)

    -- World updates
    self.world:update(dt) 

	return reward

end


