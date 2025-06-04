from PIL import Image
import os

def combine_images_vertically_and_horizontally(image_paths, output_path="combined_plot.png"):
    """
    Combines three images into one: two images horizontally on top,
    and one image below, spanning the combined width of the top two.

    Args:
        image_paths (dict): A dictionary with keys 'top_left', 'top_right', 'bottom'.
                            Values are the file paths to the images.
                            Example: {'top_left': 'plot/queue.png',
                                      'top_right': 'plot/delay.png',
                                      'bottom': 'plots/rewards.png'}
        output_path (str): The path to save the combined image.
    """
    try:
        # Load images
        img_top_left = Image.open(image_paths['top_left'])
        img_top_right = Image.open(image_paths['top_right'])
        img_bottom = Image.open(image_paths['bottom'])

        # --- Calculate sizes for combination ---

        # Resize top images to be half the width of the bottom image, maintaining aspect ratio
        # This makes the top two images fit exactly above the bottom one.
        target_top_width_per_image = img_bottom.width // 2
        
        # Resize img_top_left
        aspect_ratio_tl = img_top_left.height / img_top_left.width
        new_height_tl = int(target_top_width_per_image * aspect_ratio_tl)
        img_top_left = img_top_left.resize((target_top_width_per_image, new_height_tl), Image.Resampling.LANCZOS)

        # Resize img_top_right
        aspect_ratio_tr = img_top_right.height / img_top_right.width
        new_height_tr = int(target_top_width_per_image * aspect_ratio_tr)
        img_top_right = img_top_right.resize((target_top_width_per_image, new_height_tr), Image.Resampling.LANCZOS)

        # Ensure both top images have the same height for clean horizontal stacking
        max_top_height = max(img_top_left.height, img_top_right.height)
        img_top_left = img_top_left.resize((target_top_width_per_image, max_top_height), Image.Resampling.LANCZOS)
        img_top_right = img_top_right.resize((target_top_width_per_image, max_top_height), Image.Resampling.LANCZOS)


        # Calculate dimensions for the final combined image
        combined_top_width = img_top_left.width + img_top_right.width
        combined_top_height = max_top_height # Both top images now have this height
        
        # The overall width should be the width of the bottom image, as we scaled top images to match it.
        final_width = img_bottom.width 
        final_height = combined_top_height + img_bottom.height

        # Create a new blank image for the combined output
        combined_img = Image.new('RGB', (final_width, final_height), color = 'white') # White background

        # Paste images
        combined_img.paste(img_top_left, (0, 0)) # Top-left
        combined_img.paste(img_top_right, (img_top_left.width, 0)) # Top-right, next to top-left
        combined_img.paste(img_bottom, (0, combined_top_height)) # Bottom, below the top combined images

        # Save the combined image
        combined_img.save(output_path)
        print(f"Successfully combined images to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: One or more image files not found: {e}. Please ensure paths are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define the paths to your generated images
    # Assuming 'plot/' for queue.png and delay.png, and 'plots/' for rewards.png as per your last code snippets.
    image_paths = {
        'top_left': 'plot/queue.png',
        'top_right': 'plot/delay.png', # Assuming this is the 'delay' image
        'bottom': 'plot/rewards.png' # Assuming your rewards plot is saved here
    }

    # Create the output directory if it doesn't exist
    
    output_file = os.path.join('plot', 'fig.png')

    print("\n--- Starting Image Combination Process ---")
    combine_images_vertically_and_horizontally(image_paths, output_file)
    print("--- Image Combination Process Completed ---")