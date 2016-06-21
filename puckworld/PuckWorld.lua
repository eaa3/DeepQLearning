--[[
@Author: Ermano Arruda (exa371 at bham dot ac dot uk), June 2016

An PuckWorld to play with the PuckWorld
]]

require 'torch'
require 'math'

local PuckWorld = {}


local PPM = 64 -- Pixel per meter ratio
local F = 8*PPM



function PuckWorld:new(width,height)

    env = {}           
    env.__index = self
    env.boundaries = {}
    env.agent = {}
    env.enemy = {}
    env.food = {}
    local gx = 0.0
    local gy = 0.0 -- 9.81*64 (9.81 m/s*s)
    env.world = love.physics.newWorld(0, 0, true) 
    love.physics.setMeter(PPM) -- 1 meter in the world is equals to 64px in the scre
    env.boundaries = {{},{},{},{}}
    -- top
    env.boundaries[1].shape = love.physics.newEdgeShape( 0, 0, width, 0 )
    env.boundaries[1].body = love.physics.newBody( env.world, 0, 0, "static" )
    env.boundaries[1].fixture = love.physics.newFixture( env.boundaries[1].body, env.boundaries[1].shape, 1 )
   
    -- right
    env.boundaries[2].shape = love.physics.newEdgeShape( width, 0, width, height )
    env.boundaries[2].body = love.physics.newBody( env.world, 0, 0, "static" )
    env.boundaries[2].fixture = love.physics.newFixture( env.boundaries[2].body, env.boundaries[2].shape, 1 )
    
    -- bottom
    env.boundaries[3].shape = love.physics.newEdgeShape( 0, height, width, height )
    env.boundaries[3].body = love.physics.newBody( env.world, 0, 0, "static" )
    env.boundaries[3].fixture = love.physics.newFixture( env.boundaries[3].body, env.boundaries[3].shape, 1)
    -- left
    env.boundaries[4].shape = love.physics.newEdgeShape( 0, 0, 0, height )
    env.boundaries[4].body = love.physics.newBody( env.world, 0, 0, "static" )
    env.boundaries[4].fixture = love.physics.newFixture( env.boundaries[4].body, env.boundaries[4].shape, 1)


    env.action_space = {}
    -- right, left, up, down and nothing
    env.action_space.actions = {{x = F,y = 0}, {x = -F, y = 0}, {x = 0, y = -F}, {x = 0, y = F}, {x = 0, y = 0.0}}
    function env.action_space:sample()
    	return torch.random( #self.actions )
    end

    
    -- agent
    env.agent.shape = love.physics.newCircleShape( 25 )
    env.agent.body = love.physics.newBody( env.world, 100, 0, "dynamic" )
    env.agent.body:setLinearDamping(1.5)
    env.agent.body:setMass(10)
    -- env.agent.fixture:setFriction( .9 )
    env.agent.fixture = love.physics.newFixture( env.agent.body, env.agent.shape, 1 )
    env.agent.fixture:setRestitution(0.05)

    env.agent.sprite = love.graphics.newImage("arrow.png")
    env.agent.color = {r=14, g=47, b=193}

    env.prev_action = 1

    -- enemy
    env.enemy.position = {x=750,y=550}
    env.enemy.radius = 25
    env.enemy.effect_radius = 150
    env.enemy.speed = 0.3
    env.enemy.color = {r=193, g=47, b=14} -- red

    -- food
    env.food.radius = 15
    env.food.position = {x=love.math.random( width - env.food.radius),y=love.math.random( height - env.food.radius)}
    env.food.ttl = 500--200 -- 5 seconds to live
    env.food.time = 0
    env.food.color = {r=47, g=193, b=14}

    -- delme
    env.next_pos = 1
    --{x=30, y=30},
    env.positions = {{x=630, y=30}, {x=630, y=530}, {x=30, y=530}}

    function env.enemy:draw()
    	love.graphics.setColor(self.color.r, self.color.g, self.color.b) 

        love.graphics.circle("fill", self.position.x, self.position.y, self.radius)

        love.graphics.setColor(self.color.r, self.color.g, self.color.b, 50)
        love.graphics.circle("fill", self.position.x, self.position.y, self.effect_radius)

        love.graphics.setColor(0,0,0, 50)
        love.graphics.circle("line", self.position.x, self.position.y, self.effect_radius)
    end

    function env.enemy:update(dt)

    	local dx = env.agent.body:getX() - self.position.x
    	local dy = env.agent.body:getY() - self.position.y
    	local dir = torch.Tensor({dx,dy})
    	dir:div(torch.norm(dir)+1)

    	self.position.x = self.position.x + dir[1]*self.speed
    	self.position.y = self.position.y + dir[2]*self.speed


    end


    function env.agent:draw()
         local ori = { 0, math.rad(180), math.rad(270),  math.rad(90) } --{ 0, math.pi, 3*math.pi/4.0, -math.pi/2.0 }
         local vx, vy = self.body:getLinearVelocity()
         local speed = torch.norm(torch.Tensor({vx,vy}):double())
         local alpha = math.max(math.min(speed*0.01*255-10,255))

         love.graphics.setColor(self.color.r, self.color.g, self.color.b) 

         love.graphics.circle("fill", self.body:getX(), self.body:getY(), self.shape:getRadius())

         love.graphics.setColor(self.color.r, self.color.g, self.color.b, alpha)
         love.graphics.draw(self.sprite, self.body:getX() , self.body:getY(), ori[env.prev_action], 0.5, 0.5, self.sprite:getWidth()/2,self.sprite:getHeight()/2)
    end

    function env.agent:applyForce(fx,fy)
         self.body:applyForce(fx,fy)
    end

    function env.food:draw()
    	love.graphics.setColor(self.color.r, self.color.g, self.color.b)

        love.graphics.circle("fill", self.position.x, self.position.y, self.radius)
    end

    function env.food:update(dt)

    	self.time = self.time - 1--dt



    	if self.ttl > 0 and self.time <= 0 then
    		self.time = self.ttl
    		env.food.position.x = love.math.random( width - env.food.radius) 
    		env.food.position.y = love.math.random( height - env.food.radius) 
    	    
        end
    	
    end

    function env:getState()
        local px = self.agent.body:getX()/width - 0.5
        local py = self.agent.body:getY()/height - 0.5
        local vx, vy = self.agent.body:getLinearVelocity()

        local fpx = env.food.position.x/width - 0.5
        local fpy = env.food.position.y/height - 0.5

        local epx = env.enemy.position.x/width - 0.5
        local epy = env.enemy.position.y/height - 0.5
        --{px, py, vx, vy, fpx, fpy, epx, epy} --
        --{px, py, vx/PPM, vy/PPM, fpx - px, fpy - py}
        --{px, py, vx/PPM, vy/PPM, fpx, fpy, epx, epy}
        return torch.Tensor({px, py, vx/PPM, vy/PPM, fpx - px, fpy - py, epx - px, epy - py})
    end

    function env:getStateDim()
        return self:getState():size(1)
    end

    function env:getNActions()
        return #self.action_space.actions
    end

    function env:getActions()
        return self.action_space.actions
    end

    function env:sampleAction()
        return self.action_space:sample()
    end


    function env:step(action, dt)

        -- Agent performs action on world
        force = self.action_space.actions[action]
        self.agent:applyForce(force.x,force.y)
        self.prev_action = action

        -- World updates
        --env.world:update(0.025) 
        --env.world:update(0.04) 
        env.world:update(dt) 
        --env.world:update(dt) 
        --env.world:update(dt) -- gg


        -- Other entities update
        env.enemy:update(dt)
        env.food:update(dt)



		-- Retrieve observation, reward

		local px = self.agent.body:getX()
		local py = self.agent.body:getY()

		local fpx = env.food.position.x
		local fpy = env.food.position.y

		local epx = env.enemy.position.x
		local epy = env.enemy.position.y

		local dx = fpx - px
		local dy = fpy - py
		local d1 = math.sqrt(dx*dx + dy*dy)/math.sqrt(width*width + height*height)

		dx = epx - px
		dy = epy - py
		local d2 = math.sqrt(dx*dx + dy*dy)

		local reward = -d1

		if d2 < env.enemy.effect_radius then
			reward = reward + 2*(d2 - env.enemy.effect_radius)/env.enemy.effect_radius
		end


		obs = {r = reward}

		return obs



	end

	function env:draw()

		env.food:draw()
		env.agent:draw()
  		env.enemy:draw()
	end

    return setmetatable(env, self)

end


return PuckWorld