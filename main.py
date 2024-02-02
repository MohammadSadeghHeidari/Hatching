import os
import json
import cv2
from datetime import datetime

from hatching import hatch
from functions import remove_bg, find_levels, sculpt

from PIL import Image

def compressing(file_name):
    img = Image.open(file_name)
    img = img.resize((1000,1000),Image.LANCZOS)
    #img = img.rotate(angle=-90) #if your image(that you take by your phone) is landscape,this line should be commented
    img.save('compressed.jpg')
    return 'compressed.jpg'


def convert(file_name, output_path, params):
    with open("config.json") as jsonfile:
        config = json.load(jsonfile)

    img = remove_bg(file_name)

    img = sculpt(img)
    lvls = tuple(find_levels(img))
    print(lvls)

    
    #if params["image_scale"] == 'auto':
    #    params["image_scale"] = 800 / max(img.shape[0], img.shape[1])

    #scale_x = int(img.shape[1] * float(params["image_scale"]))
    #scale_y = int(img.shape[0] * float(params["image_scale"]))
    scale_x = 1704 #the exact value(in pixel) is 1704.5
    scale_y = 2273 #the exact value(in pixel) is 2272.7
    
    img = cv2.resize(img, (scale_x, scale_y), interpolation=cv2.INTER_LINEAR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(img.shape)
    hatch(img, output_path=output_path, circular=False, 
          hatch_angle=int(params["hatch_angle"]), board_config=config, 
          hatch_pitch=1.5, levels=lvls, blur_radius=1)


if __name__ == "__main__":
    dt = datetime.today()  
    seconds = dt.timestamp()

    os.mkdir(f"outputs/{seconds}")

    params = {
        "image_scale": 'auto', # or a float number like 0.2
        "hatch_angle": 45
    }
    convert(compressing("Taghirad.jpg"), f"outputs/{seconds}", params)