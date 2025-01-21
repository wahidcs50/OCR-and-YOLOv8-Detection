from PIL import Image

def ocr(image, processor, model):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return generated_text
