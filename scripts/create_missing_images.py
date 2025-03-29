import os
from PIL import Image, ImageDraw, ImageFont

# Configuration
OUTPUT_DIR = "app/static/all_images"
MISSING_IMAGES = [
    "image_579", "image_524", "image_999", "image_250", 
    "image_402", "image_194", "image_721", "image_72", 
    "image_558", "image_176"
]
IMAGE_SIZE = (300, 300)

def create_image(filename, size=IMAGE_SIZE):
    """Create a simple image with the filename as text"""
    # Create a new image with white background
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Use default font
    font = ImageFont.load_default()
    
    # Draw the filename in the center
    text = filename
    text_width, text_height = 150, 30  # Approximate size
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    
    # Add text
    draw.text(position, text, fill=(0, 0, 0), font=font)
    
    # Add a border
    draw.rectangle([(0, 0), (size[0]-1, size[1]-1)], outline=(200, 200, 200), width=2)
    
    return img

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create missing images
    for name in MISSING_IMAGES:
        filename = f"{name}.jpg"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"Skipping existing image: {filename}")
            continue
        
        print(f"Creating image: {filename}")
        
        # Create the image
        img = create_image(name)
        
        # Save the image
        img.save(output_path, "JPEG")
        
    print(f"Done! Created missing images in {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 