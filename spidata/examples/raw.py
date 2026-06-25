from spidata.data.registery import Registery
import logging
import pandas as pd
import cv2
from spiral.utils import setup_logging

setup_logging()

data_pack =  Registery.ot1

def general_show():
    cnt = 0
    logging.info("Images")
    for pth in data_pack.frames_path.iterdir():
        cnt+=1
        logging.info(f"{cnt} {pth}")
        if cnt ==4: break
        
    cnt = 0
    logging.info("XMLS")
    for pth in data_pack.xml_labels_path.iterdir():
        cnt+=1
        logging.info(f"{cnt} {pth}")
        if cnt ==4: 
            break
        
    logging.info("Translations")
    trans_df = pd.read_csv(data_pack.translations_path)
    logging.info(trans_df.head())

    logging.info("Txts")
    cnt = 0
    if data_pack.txt_labels_path is not None:
        for pth in data_pack.txt_labels_path.iterdir():
            cnt+=1
            logging.info(f"{cnt} {pth}")
            if cnt ==4: break
    else:
        logging.info("txt labels yok")
        
    logging.info("Test webm Image import")
    for pth in data_pack.frames_path.iterdir():
        image = cv2.imread(pth)
        if image is not None:
            logging.info(f"{pth} image başarıyla okundu: {image.shape}")
        break

def generate_labels():
    data_pack.create_txt_folder_from_xml()

general_show()
# generate_labels()