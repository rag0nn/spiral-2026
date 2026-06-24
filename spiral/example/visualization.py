from spidata.tools import get_random_sample
from spiral.struct.packets import OdObject, Result, SourcePacket, TranslationObject, SearchObject
from spiral.visualization.visualizer import Visualizer
import cv2

visualizer = Visualizer()

image, lines = get_random_sample()
H,W,_ = image.shape
objs=  []

for line in lines:
    parts = line.split(" ")
    obj = OdObject.from_xy1xy2_norm(
        0,
        int(line[0]),
        0.5,
        float(parts[1]),
        float(parts[2]),
        float(parts[3]),
        float(parts[4]),
        -1,
        0,
        (W,H)
        )
    objs.append(obj)
    
packet = SourcePacket(0, image,False,0.4,0.3,0.2,None)
translation = TranslationObject(0.5,0.2,0.4)
result = Result(packet,
       objs,
       translation,
       None)
annotated = visualizer.visualize_objs(result)

annotated = cv2.resize(annotated, (1400,900))
annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
cv2.imshow("anno", annotated)
cv2.waitKey(0)