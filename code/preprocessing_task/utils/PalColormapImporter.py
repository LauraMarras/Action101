import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Step 1: Read the .pal file
def read_pal_file(file_path):
    with open(file_path, 'r') as f:
        # Extract hex colors (assumes each line is a hex color, e.g., "#RRGGBB")
        hex_colors = [line.strip() for line in f if line.strip()]
    return hex_colors

# Step 2: Convert hex to RGB normalized values
def hex_to_rgb_normalized(hex_color):
    hex_color = hex_color.lstrip('#')  # Remove the leading '#'
    return [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]  # Normalize to [0, 1]

# Step 3: Create a colormap
def create_colormap_from_hex(hex_colors):
    rgb_colors = [hex_to_rgb_normalized(color) for color in hex_colors]
    return ListedColormap(rgb_colors)



if __name__ == '__main__':
    # Main
    pal_file_path = '/data1/Action_teresi/CCA/code/reds_and_blues.pal' # Update with the actual file path
    hex_colors = read_pal_file(pal_file_path)[1:]
    cmap = create_colormap_from_hex(hex_colors)
