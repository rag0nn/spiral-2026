from spidata.struct.registery import Registery
import logging
import pandas as pd
import cv2

data_pack =  Registery.ot25_4

def general_show():
    cnt = 0
    print("Images")
    for pth in data_pack.frames_path.iterdir():
        cnt+=1
        print(f"{cnt} {pth}")
        if cnt ==4: break
        
    cnt = 0
    print("XMLS")
    for pth in data_pack.xml_labels_path.iterdir():
        cnt+=1
        print(f"{cnt} {pth}")
        if cnt ==4: 
            break
        
    print("Translations")
    trans_df = pd.read_csv(data_pack.translations_path)
    print(trans_df.head())

    print("Txts")
    cnt = 0
    if data_pack.txt_labels_path is not None:
        for pth in data_pack.txt_labels_path.iterdir():
            cnt+=1
            print(f"{cnt} {pth}")
            if cnt ==4: break
    else:
        print("txt labels yok")
        
    print("Test webm Image import")
    for pth in data_pack.frames_path.iterdir():
        image = cv2.imread(pth)
        if image is not None:
            print(f"{pth} image başarıyla okundu: {image.shape}")
        break

def generate_labels():
    data_pack.create_txt_folder_from_xml()

# generate_labels()
# general_show()