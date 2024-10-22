import json
from pycocotools.coco import COCO

# Specify the path to your JSON file
json_file_path = '/home/tom/ws/Fast-BEV-Fusion/data/nuscenes/nuscenes_infos_train_mono3d.coco.json'

coco = COCO(json_file_path)

# # Load JSON data from file
# with open(json_file_path, 'r') as f:
#     data = json.load(f)

# # Find and print values for the image ID '2ad705761b6242c88b89344261419527'
# for entry in data['images']:
#     if entry['id'] == '2ad705761b6242c88b89344261419527':
#         print(f"Image ID: {entry['id']}")
#         print(f"File name: {entry['file_name']}")
#         print(f"Width: {entry['width']}")
#         print(f"Height: {entry['height']}")
        
#         # Add more fields as needed

#         # Assuming annotations are linked through 'id' and 'image_id'
#         annotations = [anno for anno in data['annotations'] if anno['image_id'] == entry['id']]
#         print(f"Annotations: {annotations}")

#         break  # Assuming image ID is unique, exit loop after finding the entry
# else:
#     print(f"Image ID '2ad705761b6242c88b89344261419527' not found.")


# Convert image ID string to integer if necessary (assuming it's a string)
image_id_to_search = '2ad705761b6242c88b89344261419527'

# Get annotation IDs for the specified image ID
image_id_int = coco.getImgIds(imgIds=[image_id_to_search])
annIds = coco.getAnnIds(imgIds=image_id_int)

print(f"Annotation IDs for image '{image_id_to_search}': {annIds}")