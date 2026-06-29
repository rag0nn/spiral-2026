from spidata.struct.dataset import SpiDataset
from spidata.struct.registery import Registery
from spiral.struct.packets import OdObject, Result, SourcePacket, TranslationObject, SearchObject
from spiral.visualization.visualizer import Visualizer
import cv2

visualizer = Visualizer()


dataset =SpiDataset(
    datapack=Registery.ot25_3,
    filter_missing_translations=False,
    transform=None
)
sample= dataset.get_random_sample()
image = sample["image"]
objects = sample["objects"]

H,W,_ = image.shape
objs=  []

for lbl,x,y,w,h in objects:
    obj = OdObject.from_xywh_norm(
        0,
        int(lbl),
        0.5,
        x,y,w,h,
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