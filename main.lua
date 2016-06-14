--[[
@Author: Ermano Arruda

main file
]]

require 'torch'
require 'nn'
require 'DQN'
-- disp = require ('plotting.init')

local Environment = require 'environment'

local quit = true

function love.conf(t)
  t.releases = {
    title = "PuckWorld",              -- The project title (string)
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

   env = Environment:new(800,600)
   love.window.setTitle("PuckWorld")

   love.graphics.setNewFont(12)
   love.graphics.setColor(255,255,255)
   love.graphics.setBackgroundColor(255,255,255)

   -- for saving screen shots
   love.filesystem.setIdentity('screenshots')

   agent = DQNAgent({state_dim=env:getStateDim(), n_actions=env:getNActions()})

   s = env:getState()
   agent:setInitialState(s)

   speed_idx = 1
   steps_per_iter = {1,10,50,100,500,1000,10000,0}

   worst_tderr_avg = 0

   --win = disp.plot(agent.average_reward_history, { labels={ 'Time Step', 'Reward' }, title='Average Reward' })
end

-- function draw_plot()
--   win = disp.plot(agent.average_reward_history, { win = win })
-- end

function love.draw()

  env:draw()

  local action_names = {"Right", "Left", "Up", "Down", "Nothing"}

  love.graphics.setColor(0,0,0)
  love.graphics.print('Time Step: ' .. agent.n_steps, 10, 10)
  love.graphics.print('Experience Size: ' .. agent.replay_mem.n_entries, 10, 30)
  love.graphics.print('Current TDErrror: ' .. string.format("%.4f",agent.current_tderr), 10, 50)
  if agent.average_reward ~= nil then
    love.graphics.print('Average Reward: ' .. string.format("%.4f",agent.average_reward), 10, 70)
  end
  if agent.transition_buf.r ~= nil then
    love.graphics.print('Imediate Reward: ' .. string.format("%.4f",agent.transition_buf.r), 10, 90)
  end
  if agent.recent_qs then
    love.graphics.print('Current Q values: ' .. string.format("%.4f, %.4f, %.4f, %.4f, %.4f", agent.recent_qs[1],agent.recent_qs[2],agent.recent_qs[3],agent.recent_qs[4], agent.recent_qs[5]), 10, 110)
  end
  if agent.transition_buf.a then 
      love.graphics.print('Action: ' .. action_names[agent.transition_buf.a] , 10, 130)
  end
  love.graphics.print('Episilon: ' ..   agent.episilon, 10, 150)

  --draw_plot()

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
   elseif love.keyboard.isDown("a") then
      
      agent.eta = -1
    
   elseif love.keyboard.isDown("l") then
      
      num = num + 1
      agent.eta = 0.01

   elseif love.keyboard.isDown("1") then
      
      speed_idx = 1
  elseif love.keyboard.isDown("2") then
      
      speed_idx = 2
  elseif love.keyboard.isDown("3") then
      
      speed_idx = 3
  elseif love.keyboard.isDown("4") then
      
      speed_idx = 4
  elseif love.keyboard.isDown("5") then
      
      speed_idx = 5
  elseif love.keyboard.isDown("6") then
      
      speed_idx = 6
  elseif love.keyboard.isDown("7") then
      
      speed_idx = 7
  elseif love.keyboard.isDown("s") then
      
      agent:save("qnet.7z")
      print("Saved agent.")
  elseif love.keyboard.isDown("d") then
      
      agent:load("qnet.7z")
      print("Loaded agent.")
      agent.eta = -1 -- do not learn
  elseif love.keyboard.isDown("p") then
      
      speed_idx = 8
      print("Paused!")
   end


    if action ~= -1 then
        env:step(action,dt)
    end


    for i = 1, steps_per_iter[speed_idx] do
      s = env:getState()

      if agent.eta > 0 then
        obs = env:step(agent:act(torch.Tensor(s),false),dt)
        agent:perceive(obs)
      else
        obs = env:step(agent:act(torch.Tensor(s),true),dt)
        agent:perceive(obs)
      end
    end

    if speed_idx == 7 then
      speed_idx = 6
    end



    -- for k, v in pairs( obs ) do
    --     print(k, v)
    -- end

   -- if love.mouse.isDown(1) then
   --  print("Saving screenshot to " .. love.filesystem.getSaveDirectory())
   --  local screenshot = love.graphics.newScreenshot()
   --  screenshot:encode('png',  os.time() .. '.png')
   --  --love.filesystem.write("pic.png", screenshot)
   --  --torch.save('test.t7', {hello=123, world=torch.rand(1,2,3)})
   -- end





end



function love.mousepressed(x, y, button, istouch)
   if button == 1 then
      --imgx = x - image:getWidth()/2 -- move image to where mouse clicked
      --imgy = y - image:getWidth()/2 
   end
end

function love.mousereleased(x, y, button, istouch)
   if button == 1 then
      --fireSlingshot(x,y) -- this totally awesome custom function is defined elsewhere
   end
end


function love.quit()
    print("See you in a bit!")
    return false
end




