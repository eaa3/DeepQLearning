--[[
@Author: Ermano Arruda

Food can be either poisonous or not
]]

require 'torch'

local Food = torch.class("Food")


function Food:__init(is_poison, world,ppm,vm)
	self.vm = vm -- force magnitude
	self.ppm = ppm
	self.shape = love.physics.newCircleShape( 20 )
    self.body = love.physics.newBody( world, 100, 0, "dynamic" )
    self.body:setLinearDamping(0)
    self.body:setMass(0.01)
    self.fixture = love.physics.newFixture( self.body, self.shape, 1 )
    self.fixture:setRestitution(1.0)
    self.fixture:setFriction(0)

    self.is_poison = is_poison
    self.is_active = true

    if is_poison then
    	self.color = {r=47, g=193, b=14}
    else
    	self.color = {r=193, g=47, b=14}
    end

    self:reinitRandom()

end

function Food:toggleActive()
	self.is_active = not self.is_active
end

function Food:reinitRandom()

	self:setRandomVelocity()
    self:setRandomPos()

end

function Food:setRandomVelocity()
	local rf = torch.randn(2)
    rf = rf:div(rf:norm())
    local fx = rf[1]*self.ppm*self.vm
    local fy = rf[2]*self.ppm*self.vm

    self.body:setLinearVelocity(fx,fy)
    --self.body:applyForce(fx,fy)
end

function Food:setRandomPos()

    local rp = torch.rand(2)
    local px = rp[1]*love.graphics.getWidth()
    local py = rp[2]*love.graphics.getHeight()

    self.body:setPosition(px, py)
end

function Food:draw()
	love.graphics.setColor(self.color.r, self.color.g, self.color.b) 
	love.graphics.circle("fill", self.body:getX(), self.body:getY(), self.shape:getRadius())
end
