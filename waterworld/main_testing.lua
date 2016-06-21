--[[
@Author: Ermano Arruda (exa371 at bham dot ac dot uk), June 2016

main file
]]

require 'torch'
require 'nn'
require 'utils'


function love.conf(t)
  t.releases = {
    title = "Test",              -- The project title (string)
    package = nil,            -- The project command and package name (string)
    loveVersion = nil,        -- The project LÖVE version
    version = nil,            -- The project version
    author = nil,             -- Your name (string)
    email = nil,              -- Your email (string)
    description = nil,        -- The project description (string)
    homepage = nil,           -- The project homepage (string)
    identifier = nil,         -- The project Uniform Type Identifier (string)
    releaseDirectory = nil,   -- Where to store the project releases (string)
  }
end


function love.load()

  -- Setting seed
  torch.manualSeed(0)

  love.window.setTitle("Test")

  love.graphics.setNewFont(12)
  love.graphics.setColor(255,255,255)
  love.graphics.setBackgroundColor(255,255,255)

   ball = {}
   ball.x = 300
   ball.y = 300
   ball.radius = 100
   ball.segments = 20

   intersect = {}
   intersect.x1 = 0
   intersect.y1 = 0
   intersect.x2 = 0
   intersect.y2 = 0
   intersect.x3 = 0
   intersect.y3 = 0
   intersect.t1 = -1
   intersect.t2 = -1
   intersect.t3 = -1
   intersect.radius = 5



   line = {}
   line.dir_idx = 0
   line.dirs = torch.cat(torch.cos(torch.linspace(0,2*math.pi,10)),-torch.sin(torch.linspace(0,2*math.pi,10)),2)
   line.dir = line.dirs:narrow(1,1,1):view(-1)--torch.Tensor({math.cos(math.pi/4),-math.sin(math.pi/4)})
   line.pos = torch.zeros(2)

   function line:getPoints()
   	local x1 = line.pos[1]
   	local y1 = line.pos[2]
   	local x2 = x1 + line.dir[1]*100
   	local y2 = y1 + line.dir[2]*100

   	return {x1,y1,x2,y2}
   end

end

function love.draw()

	love.graphics.circle( 'line', ball.x, ball.y, ball.radius, ball.segments )

	love.graphics.setColor(100,100,100)
	love.graphics.line( line:getPoints() )

	love.graphics.circle( 'line', ball.x, ball.y, ball.radius, ball.segments )

	love.graphics.circle( 'fill', intersect.x1, intersect.y1, intersect.radius, 10 )
	love.graphics.circle( 'fill', intersect.x2, intersect.y2, intersect.radius, 10 )

	love.graphics.circle( 'fill', intersect.x3, intersect.y3, intersect.radius, 10 )
	love.graphics.line( line.pos[1], line.pos[2], intersect.x3, intersect.y3)
end


function love.update(dt)


	action = -1

   if love.keyboard.isDown("up") then
      
      action = 3
   elseif love.keyboard.isDown("down") then
      
      action = 4 
   elseif love.keyboard.isDown("right") then
      
      action = 1
   elseif love.keyboard.isDown("left") then
      
      action = 2
   end

   if action > 0 then
   	--local r = env:step(action,dt)
   	--print("Reward: " .. r)
   end

end



function love.quit()
    print("See you in a bit!")
    return false
end

function love.keypressed( key )

   if key == 'return' then
      text = "RETURN has been pressed!"
   end

   if key == 'r' then
   	line.dir_idx = (line.dir_idx+1)%line.dirs:size(1) + 1
   	line.dir = line.dirs:narrow(1,line.dir_idx,1):view(-1):clone()
   	print(line.dir)
   end

end




function love.mousepressed(x, y, button, istouch)
   if button == 1 then

   	  local pts = line:getPoints()
      out = utils.computeIntersectLine(line.pos[1],line.pos[2],line.dir[1],line.dir[2],800,0,1,0)--utils.computeIntersect2(pts[1],pts[2],pts[3],pts[4],ball.x,ball.y,ball.radius)

      if out then
      	intersect.x1 = out.x1
      	intersect.y1 = out.y1
      	intersect.x2 = out.x2
      	intersect.y2 = out.y2
      	intersect.x3 = out.x3
      	intersect.y3 = out.y3

      	print(out.x3 .. ", " .. out.y3 .. ", " .. out.t3)
      end
      
   end
end

function love.mousemoved( x, y, dx, dy, istouch )
	line.pos[1] = x
	line.pos[2] = y
end




