--[[
@Author: Ermano Arruda

main file
]]

require 'torch'
require 'nn'
require 'WaterWorld'
require 'rl.DQN'
Plot = require ('plotting.slide_plot')


function love.conf(t)
  t.releases = {
    title = "WaterWorld",              -- The project title (string)
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

  -- Setting seed
  torch.manualSeed(0)

   env = WaterWorld(800,600)
   love.window.setTitle("WaterWorld")

   love.graphics.setNewFont(12)
   love.graphics.setColor(255,255,255)
   love.graphics.setBackgroundColor(255,255,255)

   opt = {state_dim=env:getStateDim(), 
          n_actions=env:getNActions(),
          batch_size=100,
          lr = 0.002,
          lr_endt = 1000,
          n_hidden_units = 100
          }
   agent = DQNAgent(opt)

   s = env:getState()
   agent:setInitialState(s)

   if #args > 1 then
    agent:load(args[2])
    agent.eta = -1
   end

   speed_idx = 1
   steps_per_iter = {1,2,10,50,100,500,1000,10000,0}


end

function speed_control()
  if love.keyboard.isDown("1") then
      
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
  elseif love.keyboard.isDown("8") then
      
      speed_idx = 8
  elseif love.keyboard.isDown("p") then
      
      speed_idx = 9
      print("Paused!")
   end
end

function save_control()
  if love.keyboard.isDown("s") then
      
      agent:save("qnet.7z")
      print("Saved agent.")
  elseif love.keyboard.isDown("d") then
      
      agent:load("qnet.7z")--("qnet_night.t7")
      print("Loaded agent.")
      --agent.eta = -1 -- do not learn
  end
end

function learning_control()
  if love.keyboard.isDown("a") then
      
      agent.eta = -1
    
   elseif love.keyboard.isDown("l") then
      
      agent.eta = 0.01
   end
end

function agent_control()
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
      
      env.world:update(dt)
   end

   if action > 0 then
    env:getState()
    local r = env:step(action,dt)
    --print("Reward: " .. r)
   end
end

function controls()
  speed_control()
  save_control()
  learning_control()
  agent_control()
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

  draw_plot()

  love.graphics.print('Food/TotalEaten Ratio: ' .. (env.n_eaten_food/(env.n_eaten_poison+env.n_eaten_food)), 10, 10)

end


function love.update(dt)


    controls()


    for i = 1, steps_per_iter[speed_idx] do
      local s = env:getState()

      if agent.eta > 0 then
        obs = env:step(agent:act(s,false),dt)
        --obs = env:step(env.action_space:sample(),dt)
        agent:perceive(obs)
      else
        obs = env:step(agent:act(s,true),dt)
        agent:perceive(obs)
      end
    end

    if speed_idx == 8 then
      speed_idx = 7
    end

    collectgarbage()

   

end



function love.quit()
    print("See you in a bit!")
    return false
end

function love.keypressed( key )

   if key == 'return' then
      text = "RETURN has been pressed!"
   end

end





