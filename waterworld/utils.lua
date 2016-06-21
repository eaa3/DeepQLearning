--[[
@Author: Ermano Arruda

Utilities to calculate intersection of lines with circles and other lines
]]

utils = {}

function utils.computeIntersect(lpx,lpy,a,b,cx,cy,r)
	local intersect_points = nil

	local x0 = lpx - cx
    local y0 = lpy - cy
    local r2 = r*r
    local den = a*a + b*b -- assuming (a,b) is a normalised vector

    local delta = 2*a*x0*b*y0 + den*r2 - a*a*x0*x0 - b*b*y0*y0
    if delta >= 0 then
    	intersect_points = {}

      	local u = -a*x0 - b*y0 
      	local t1 = (u + math.sqrt(delta))/den
      	local t2 = (u - math.sqrt(delta))/den


      	intersect_points.x1 = lpx + a*t1
      	intersect_points.y1 = lpy + b*t1
      	intersect_points.t1 = t1

      	intersect_points.x2 = lpx + a*t2
      	intersect_points.y2 = lpy + b*t2
      	intersect_points.t2 = t2

      	--print(t1 .. ", " .. t2)
    end


    return intersect_points

end

local function sign(x)
	if x < 0 then
		return -1
	else
		return 1
	end
end

function utils.computeIntersect2(x1,y1,x2,y2,cx,cy,r)
	local intersect_points = nil

	local x_1 = x1 - cx
	local x_2 = x2 - cx

	local y_1 = y1 - cy
	local y_2 = y2 - cy

	local dx = x_2 - x_1
	local dy = y_2 - y_1

	local dr = math.sqrt(dx*dx + dy*dy)

	local D = x_1*y_2 - x_2*y_1

    local delta = r*r*dr*dr - D*D


    if delta >= 0 then
    	intersect_points = {}


      	intersect_points.x1 = (D*dy + sign(dy)*dx*math.sqrt(delta))/(dr*dr)
      	intersect_points.y1 = (-D*dx + math.abs(dy)*math.sqrt(delta))/(dr*dr)

      	local dx1 = (intersect_points.x1 - x_1)
      	local dy1 = (intersect_points.y1 - y_1)
      	intersect_points.t1 = math.sqrt(dx1*dx1 + dy1*dy1)*sign(dx*dx1+dy*dy1)

      	intersect_points.x2 = (D*dy - sign(dy)*dx*math.sqrt(delta))/(dr*dr)
      	intersect_points.y2 = (-D*dx - math.abs(dy)*math.sqrt(delta))/(dr*dr)
      	
      	dx1 = (intersect_points.x2 - x_1)
      	dy1 = (intersect_points.y2 - y_1)
      	intersect_points.t2 = math.sqrt(dx1*dx1 + dy1*dy1)*sign(dx*dx1+dy*dy1)

  	
  		intersect_points.x1 = intersect_points.x1 + cx
  		intersect_points.y1 = intersect_points.y1 + cy

  		intersect_points.x2 = intersect_points.x2 + cx
  		intersect_points.y2 = intersect_points.y2 + cy

      	-- print(intersect_points.x1 .. ", " .. intersect_points.y1 .. ", " .. intersect_points.t1 )
      	-- print(intersect_points.x2 .. ", " .. intersect_points.y2 .. ", " .. intersect_points.t2 )
      	-- print("-----")
    end


    return intersect_points

end

function utils.computeIntersectLine(px,py,vx,vy,qx,qy,a,b)

	local d = a*vx + b*vy

	if d == 0 then
		return nil
	end

	local t = (-a*px - b*py + a*qx + b*qy)/d


	local intersect = {}

	local xp = px + vx*t
	local yp = py + vy*t


	intersect.x1 = 0
	intersect.y1 = 0
	intersect.t1 = -1

	intersect.x2 = 0
	intersect.y2 = 0
	intersect.t2 = -1

	intersect.x3 = xp
	intersect.y3 = yp
	intersect.t3 = t


	return intersect




end

return utils