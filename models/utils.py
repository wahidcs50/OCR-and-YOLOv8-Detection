from trocr_processing import ocr

def map_players_to_positions(saved_images, processor, model):
    player_names = []
    player_positions = []

    for player_name_info, player_position_info in zip(saved_images["player_name"], saved_images["player_position"]):
        player_name_image = Image.open(player_name_info["filename"]).convert('RGB')
        player_position_image = Image.open(player_position_info["filename"]).convert('RGB')

        player_name_text = ocr(player_name_image, processor, model).strip()
        player_position_text = ocr(player_position_image, processor, model).strip()

        player_names.append(player_name_text)
        player_positions.append(player_position_text)

    return dict(zip(player_names, player_positions))
