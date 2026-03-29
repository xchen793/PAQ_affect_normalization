from PIL import Image
import os

input_dir = "./"
output_dir = "./images_compressed"
os.makedirs(output_dir, exist_ok=True)

for f in os.listdir(input_dir):
    if f.lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(os.path.join(input_dir, f))

        img = img.convert("RGB")

        img.thumbnail((1024, 1024))

        out = os.path.splitext(f)[0] + ".png"
        img.save(
            os.path.join(output_dir, out),
            "PNG",
            optimize=True
        )

print("Done")
