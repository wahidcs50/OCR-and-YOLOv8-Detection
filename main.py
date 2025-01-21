import os
from dotenv import load_dotenv
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from models.detection_model import process_detections
from models.utils import map_players_to_positions
from models.utils import getting_detections
import cv2
import torch
from pathlib import Path

load_dotenv()


image_path = os.getenv('IMAGE_PATH')

def main():

    image = cv2.imread(image_path)
    
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-str')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-str')

   
    model_path = os.getenv('YOLOV8_MODEL_PATH')  # Path to your finetuned YOLOv8 model
    detection_model =YOLO(model_path)


    print("Performing detection...")
    detections = detection_model(detection_model, image)

    print("Processing detections...")
    saved_images = process_detections(image, detections)

    print("Mapping player names to positions...")
    player_mapping = map_players_to_positions(saved_images, processor, model)
    
    if player_mapping:
        print("Mapped Player Names to Positions:")
        for player_name, player_position in player_mapping.items():
            print(f"{player_name}: {player_position}")
    else:
        print("No detections found.")

if __name__ == "__main__":
    main()
