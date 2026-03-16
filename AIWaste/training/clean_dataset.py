from PIL import Image
import os

dataset_path = "../dataset/dataset_waste"   # if running from training folder

deleted_count = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)

        if file.startswith("."):
            print(f"Removing hidden file: {file_path}")
            os.remove(file_path)
            deleted_count += 1
            continue

        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            print(f"Deleting bad file: {file_path} -> {e}")
            os.remove(file_path)
            deleted_count += 1

print(f"\nDone. Removed {deleted_count} bad/hidden files.")
