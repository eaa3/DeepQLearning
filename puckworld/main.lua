--[[
@Author: Ermano Arruda (exa371 at bham dot ac dot uk), June 2016

main file
]]

require 'torch'
require 'nn'
require 'rl.DQN'
Plot = require ('plotting.slide_plot')

local Environment = require 'PuckWorld'

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


function love.load(args)

   -- routine = coroutine.create(function ()
   --         os.execute("source plotting/start_server.bash")
   --       end)
   -- coroutine.resume(routine)

  -- Setting seed
  torch.manualSeed(0)

   env = Environment:new(800,600)
   love.window.setTitle("PuckWorld")

   love.graphics.setNewFont(12)
   love.graphics.setColor(255,255,255)
   love.graphics.setBackgroundColor(255,255,255)

   -- for saving screen shots
   love.filesystem.setIdentity('screenshots')

   opt = {state_dim=env:getStateDim(), 
          n_actions=env:getNActions(),
          batch_size=35,
          lr = 0.01,
          lr_endt = 500000,
          n_hidden_units = 100,
          update_type = "rmsprop"
   }

   agent = DQNAgent({state_dim=env:getStateDim(), n_actions=env:getNActions()})

   s = env:getState()
   agent:setInitialState(s)

   speed_idx = 1
   steps_per_iter = {1,10,50,100,500,1000,10000,0}

   if #args > 1 then
    agent:load(args[2])
    agent.eta = -1
   end

   win_average_reward = nil
   win_tderr = nil


end

function draw_plot()

  if agent.average_reward_hist:count() > 0 then

    win_average_reward = Plot.plot(agent.average_reward_hist,win_average_reward,{labels={"Time Step","Average Reward"},title="Reward Progress"})
  end

  if agent.tderr_hist:count() > 0 then

    win_tderr = Plot.plot(agent.tderr_hist,win_tderr,{labels={"Time Step","TD Error"},title="TD Error Progress"})
  end


end

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
  if agent.transition_buf and agent.transition_buf.r ~= nil then
    love.graphics.print('Imediate Reward: ' .. string.format("%.4f",agent.transition_buf.r), 10, 90)
  end
  -- if agent.recent_qs then
  --   love.graphics.print('Current Q values: ' .. string.format("%.4f, %.4f, %.4f, %.4f, %.4f", agent.recent_qs[1],agent.recent_qs[2],agent.recent_qs[3],agent.recent_qs[4], agent.recent_qs[5]), 10, 110)
  -- end
  if agent.transition_buf and agent.transition_buf.a then 
      love.graphics.print('Action: ' .. action_names[agent.transition_buf.a] , 10, 130)
  end
  love.graphics.print('Episilon: ' ..   agent.episilon, 10, 150)

  draw_plot()

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
   elseif love.keyboard.isDown("0") then
      
      action = 5 
   elseif love.keyboard.isDown("a") then
      
      agent.eta = -1
    
   elseif love.keyboard.isDown("l") then
      
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
      
      agent:load("qnet_night.t7")
      print("Loaded agent.")
      agent.eta = -1 -- do not learn
  elseif love.keyboard.isDown("p") then
      
      speed_idx = 8
      print("Paused!")
   end


    if action ~= -1 then
        s = env:getState()
        a = agent:act(torch.Tensor(s))
        env:step(action,dt)
        agent:perceive(obs)
    end


    for i = 1, steps_per_iter[speed_idx] do
      s = env:getState()

      if agent.eta > 0 then
        obs = env:step(agent:act(torch.Tensor(s),false),dt)
        --obs = env:step(env.action_space:sample(),dt)
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




