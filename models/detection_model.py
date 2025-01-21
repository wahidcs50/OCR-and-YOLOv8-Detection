import cv2

def process_detections(image, detections, conf_threshold=0.7):
    saved_images = {"player_name": [], "player_position": []}

    if detections.xyxy.shape[0] > 0:
        for i, (bbox, class_id, conf) in enumerate(zip(detections.xyxy, detections.class_id, detections.confidence)):
            if conf > conf_threshold:
                xmin, ymin, xmax, ymax = map(int, bbox)
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(image.shape[1], xmax)
                ymax = min(image.shape[0], ymax)

                filename = f'cropped_image_class_{class_id}_box_{i}.jpg'
                cropped_image = image[ymin:ymax, xmin:xmax]
                cv2.imwrite(filename, cropped_image)

                key = "player_name" if class_id == 0 else "player_position"
                saved_images[key].append({"filename": filename, "ymin": ymin})

        saved_images["player_name"].sort(key=lambda x: x["ymin"])
        saved_images["player_position"].sort(key=lambda x: x["ymin"])

    return saved_images
