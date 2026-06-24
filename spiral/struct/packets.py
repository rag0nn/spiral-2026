import numpy as np
import logging
from .utils import check_obj_overflow
from beartype import beartype

@beartype
class OdObject:
    
    def __init__(self,
        id_:int,
        cls:int,
        landing_status:int,
        motion_status:int,
        top_left_x:int,# piksel         
        top_left_y:int,# piksel         
        bottom_right_x:int, # piksel        
        bottom_right_y:int,# piksel
        ):
        self.id_ = id_
        self.cls = cls
        self.landing_status = landing_status
        self.motion_status = motion_status
        self.top_left_x = top_left_x            
        self.top_left_y = top_left_y             
        self.bottom_right_x = bottom_right_x           
        self.bottom_right_y = bottom_right_y
        
    def check_attribute_values(self):
        valid_cls = [0,1,2,3]
        valid_landing_status = [-1,0,1]
        valid_motion_status = [-1,0,1]
        if not self.cls in valid_cls:
            logging.warning(f"CheckWarn: Obje classı geçersiz değerde: {self.cls} {type(self.cls)}, Bunlardan biri olmalı: {valid_cls} ")
        if not self.landing_status in valid_landing_status:
            logging.warning(f"CheckWarn: Obje landing statusu geçersiz değerde: {self.landing_status} {type(self.landing_status)}, Bunlardan biri olmalı: {valid_landing_status} ")
        if not self.motion_status in valid_motion_status:
            logging.warning(f"CheckWarn: Obje motion statusu geçersiz değerde: {self.motion_status} {type(self.motion_status)}, Bunlardan biri olmalı: {valid_motion_status} ")

        # class türüne göre checkler
        if self.cls == 0:
            if not self.landing_status == -1:
                logging.warning(f"CheckWarn: {self.cls} class öğesi 'landing_status' için bu değeri alamaz: {self.landing_status}")
            if not self.motion_status in [0,1]:
                logging.warning(f"CheckWarn: {self.cls} class öğesi 'motion_status' için bu değeri alamaz: {self.motion_status}")
        elif self.cls == 1:
            if not self.landing_status == -1:
                logging.warning(f"CheckWarn: {self.cls} class öğesi 'landing_status' için bu değeri alamaz: {self.landing_status}")
            if not self.motion_status  == -1:
                logging.warning(f"CheckWarn: {self.cls} class öğesi 'motion_status' için bu değeri alamaz: {self.motion_status}")
        elif self.cls == 2 or self.cls == 3:
            if not self.landing_status in [0,1]:
                logging.warning(f"CheckWarn: {self.cls} class öğesi 'landing_status' için bu değeri alamaz: {self.landing_status}")
            if not self.motion_status == -1:
                logging.warning(f"CheckWarn: {self.cls} class öğesi 'motion_status' için bu değeri alamaz: {self.motion_status}")
             
    def __str__(self):
        return (f"OdObject(id_={self.id_}, cls={self.cls}, "
                f"landing_status={self.landing_status}, motion_status={self.motion_status}, "
                f"top_left=({self.top_left_x}, {self.top_left_y}), "
                f"bottom_right=({self.bottom_right_x}, {self.bottom_right_y}))")
    
@beartype
class TranslationObject:
    
    def __init__(self,
        x: float,
        y: float,
        z: float,
        ):
        self.x = x # metre
        self.y = y # metre
        self.z = z # metre

    def __str__(self):
        return f"TranslationObject(x={self.x}, y={self.y}, z={self.z})"
    
@beartype
class SearchObject:
    
    def __init__(self,
        id_:int,
        top_left_x:int, # piksel         
        top_left_y:int, # piksel         
        bottom_right_x:int, # piksel         
        bottom_right_y:int, # piksel  
        ):
        self.id_ = id_
        self.top_left_x = top_left_x            
        self.top_left_y = top_left_y             
        self.bottom_right_x = bottom_right_x           
        self.bottom_right_y = bottom_right_y

    def __str__(self):
        return (f"SearchObject(id_={self.id_}, "
                f"top_left=({self.top_left_x}, {self.top_left_y}), "
                f"bottom_right=({self.bottom_right_x}, {self.bottom_right_y}))")

@beartype
class SourcePacket:
    
    def __init__(self,
        id_:int,
        original_image:np.ndarray,
        gps_health_status:bool,
        gps_translation_x:float = None, # metre
        gps_translation_y:float = None, # metre        
        gps_translation_z:float = None, # metre         
        search_image:np.ndarray = None
        ):
        # inputs
        self.id_ = id_
        self.original_image = original_image
        self.original_shape = original_image.shape
        self.gps_health_status = gps_health_status
        self.gps_translation_x = gps_translation_x 
        self.gps_translation_y = gps_translation_y
        self.gps_translation_z = gps_translation_z
        self.search_image = search_image

    def __str__(self):
        return (f"SourcePacket(id_={self.id_}, "
                f"original_shape={self.original_shape}, "
                f"gps_health_status={self.gps_health_status}, "
                f"gps_translation=({self.gps_translation_x}, {self.gps_translation_y}, {self.gps_translation_z}), "
                f"search_image={'None' if self.search_image is None else 'set'})")

@beartype
class Result:
    
    def __init__(self,
        packet: SourcePacket,
        od_objects: list[OdObject],
        translation_object: TranslationObject,
        search_object: SearchObject = None
        ):
        self.packet = packet
        self.od_objects = od_objects
        self.translation_object = translation_object
        self.search_object = search_object

    def __str__(self):
        return (f"Result(packet={self.packet}, "
                f"od_objects={self.od_objects}, "
                f"translation_object={self.translation_object}, "
                f"search_object={self.search_object})")

    def check(self):
        # check valid labels
        for obj in self.od_objects:
            obj.check_attribute_values()
        
        # check object position overflows
        h,w = self.packet.original_shape[:2]
        
        for obj in self.od_objects:
            if not check_obj_overflow(h,w,
                obj.top_left_x,
                obj.top_left_y,
                obj.bottom_right_x,
                obj.bottom_right_y):
                raise ValueError(f"Obje resmin dışına taşıyor. {str(obj)} image: (0,0)({w}{h})")
        
        if not check_obj_overflow(h,w,
            self.search_object.top_left_x,
            self.search_object.top_left_y,
            self.search_object.bottom_right_x,
            self.search_object.bottom_right_y,
            ): 
            raise ValueError(f"Obje resmin dışına taşıyor. {str(self.search_object)} image: (0,0)({w}{h})")
                 