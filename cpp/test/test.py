import numpy as np
from cloudvolume import view
import zfpy

data = open("0-128_0-128_0-10-1.zfp", "rb").read()
img = zfpy.decompress_numpy(data)

print(img)
view(img, port=8081)