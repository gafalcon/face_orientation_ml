import os
from os.path import join
import sys

filepath = "fernando"
filename = "fernando.mp4"
up = down = left = right = center = tv = 1

splits = []

splits.append((1,20,"center"))
splits.append((26,38,"right"))
splits.append((40,56,"left"))
splits.append((58,77,"up"))
splits.append((80,98,"down"))
splits.append((101,120,"tv"))
splits.append((125,135,"tv"))
splits.append((138,158,"center"))
splits.append((160,166,"right"))
splits.append((168,176,"left"))
splits.append((177,182,"up"))
splits.append((184,208,"down"))

for (init,end,name) in splits:
    if name == "up":
        res = "up"+str(up)+".mp4"
        up +=1
    elif name == "center":
        res = "center"+str(center)+".mp4"
        center += 1
    elif name == "down":
        res = "down"+str(down)+".mp4"
        down += 1
    elif name == "left":
        res = "left"+str(left)+".mp4"
        left += 1
    elif name == "right":
        res = "right"+str(right)+".mp4"
        right += 1
    elif name == "tv":
        res = "tv"+str(tv)+".mp4"
        tv +=1
    os.system('ffmpeg -i {} -ss {} -c copy -to {} {}'.format(
        filename,
        init,
        end,
        join(filepath,res)))
