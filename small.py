import time

from PIL import Image#引入PIL库中的Image类
import os#引入os 模块
save_folder = "/home/ps/DiskA/project/dxf-pdf/english"
im_folder = "/home/ps/DiskA/project/dxf-pdf/usbRgbPicSave-225_bmp"
s = time.time()
for im_file in os.listdir(im_folder):
    picture = os.path.join(im_folder,im_file)
    image=Image.open(picture)
    size=os.path.getsize(picture)/1024
    width,height=image.size
    ima= os.path.join(save_folder, im_file)
    while True:
        if size>10:
            width,height=round(width*0.9),round(height*0.9)#去掉浮点，防报错
            image=image.resize((width,height),Image.ANTIALIAS)
            image.save(ima)
            size=os.path.getsize(ima)/1024
        else:
            break
        # print(size)
print(time.time()-s)
