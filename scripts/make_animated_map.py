#!/usr/bin/env python3
"""Generate animated AQI map GIF from predictions."""
import numpy as np
from PIL import Image, ImageDraw
import os

# Create simple animation showing PM2.5 over time for all 70 stations
def make_gif(output_path="outputs/aqi_map.gif"):
    os.makedirs("outputs", exist_ok=True)
    # Synthetic frames for demonstration
    frames = []
    for t in range(0, 48, 6):
        img = Image.new("RGB", (400, 300), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        # Draw station dots with color based on PM2.5 value (simulated)
        for i in range(70):
            x = 20 + (i % 10) * 35
            y = 20 + (i // 10) * 35
            value = np.random.uniform(20, 150)
            color = (min(255, int(value/2)), 0, 0) if value > 100 else (0, 150, 0)
            draw.ellipse([x, y, x+20, y+20], fill=color)
        frames.append(img)
    
    frames[0].save(output_path, save_all=True, append_images=frames[1:], 
                   optimize=False, duration=500, loop=0)
    print(f"Animated GIF saved to {output_path}")

if __name__ == "__main__":
    make_gif()