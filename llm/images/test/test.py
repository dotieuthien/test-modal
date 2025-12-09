import os
import sys
from PIL import Image


def process_images():
    # Directory containing the images
    directory = os.path.dirname(os.path.abspath(__file__))

    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_files = []

    for file in os.listdir(directory):
        # Check if the file is an image
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in image_extensions and file != 'test.py':
            image_files.append(file)

    # Process each image file
    for index, file in enumerate(image_files, 1):
        try:
            # Open image
            img_path = os.path.join(directory, file)
            img = Image.open(img_path)

            # Save as PNG with index
            output_filename = f"{index}.png"
            output_path = os.path.join(directory, output_filename)

            # Convert to RGB if the image has an alpha channel (RGBA)
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            img.save(output_path, 'PNG')
            print(f"Saved {file} as {output_filename}")

        except Exception as e:
            print(f"Error processing {file}: {e}")


if __name__ == "__main__":
    process_images()
