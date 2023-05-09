from PIL import Image
import os

folder_path = "Phase2\icons"

min_size = float('inf') # Set initial value to infinity

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        with Image.open(os.path.join(folder_path, filename)) as img:
            width, height = img.size
            if width < min_size and height < min_size:
                min_size = min(width, height)

print(f"The smallest size is {min_size} pixels.")