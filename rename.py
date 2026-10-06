from pathlib import Path

folder = Path(r"C:\Users\kavan\OneDrive\Desktop\IIITH\sem3\MR\Camera-Callibration\data\set_c")
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

images = sorted(
    file for file in folder.iterdir()
    if file.is_file() and file.suffix.lower() in image_extensions
)

# Temporary names prevent filename collisions
temporary_files = []
for index, image in enumerate(images, start=1):
    temporary = image.with_name(f"__temp_{index}{image.suffix}")
    image.rename(temporary)
    temporary_files.append(temporary)

for index, temporary in enumerate(temporary_files, start=1):
    temporary.rename(folder / f"C_{index}.jpeg")

print(f"Renamed {len(images)} images.")