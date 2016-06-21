--[[
@Author: Ermano Arruda

Useful functions for plotting
]]

-- disp = require ('plotting.init')
disp = require 'display'
require 'plotting.CircularBuffer'
require 'socket'


local Plot = {}

function Plot.makePlotData(values)
	local data = {}
	for i=1, values:size(1) do
		table.insert(data,{i,values[i]})
	end

	return data
end

function Plot.plot(buffer,win,opt)
	local data = Plot.makePlotData(buffer:queryAll())
	
	if win == nil then
		win = disp.plot(data,{labels=opt.labels, title=opt.title})
	else
		disp.plot(data,{win=win})
	end

	return win
end

return Plot



-- local buf = CircularBuffer(50)
-- local win = nil


-- for i=1,500 do

-- 	buf:insert(torch.rand(1))
	
-- 	win = plot(buf,win)

-- 	socket.select(nil, nil, 0.2); -- this is just sleeping for specified timeout (0.2 time units)

	
-- end
