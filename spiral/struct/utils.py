def check_obj_overflow(h:int, w:int, obj_x1:int, obj_y1:int, obj_x2:int, obj_y2:int)->bool:
    return (
        0 <= obj_x1 < obj_x2 <= w and
        0 <= obj_y1 < obj_y2 <= h
    )