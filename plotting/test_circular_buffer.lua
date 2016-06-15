require 'CircularBuffer'


list = CircularBuffer(5)

for i=1,10 do
	list:insert(i)

	print("--")
	for j = 1, list:count() do
		print(list:get(j))
	end
	--print(list:queryAll())
end




-- data = {}
-- for i=1,15 do
--   table.insert(data, { i, math.random(), math.random() * 2 })
-- end

-- win = disp.plot(data, { labels={ 'position', 'a', 'b' }, title='progress' })