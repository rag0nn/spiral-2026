from .config.spiral import SpiralConfig
from .utils import time_monitor ,setup_logging
from .struct.packets import *

class Spiral:
    
    def __init__(self, config:SpiralConfig):
        setup_logging()
        self.config = config
        self.counter = -1
    
    def __call__(self,  input_pack: SourcePacket)->Result:
        return self.detect(input_pack)
    
    @time_monitor
    def detect(self, input_pack: SourcePacket)->Result:
        return Result