from imageio.v2 import imread
import imageio.v3 as iio

from simulation import T

output_file = "simulation.gif"
initial = 0
final = T

# Check if the output file already
if output_file in os.listdir():
    print("Output file already exists. Overwrite? (y/n)")
    if input().lower() != "y":
        print("Exiting...")
        exit()

states = [
    f"figure_{t}.png"
    for t in range(initial, final + 1)
]

states.append("figure_convergence.png")

# Create a GIF from the images
print("Creating simulation GIF...")
# Read all the image files
images = [imread(f"figures/{state}") for state in states]

# Save the images as a GIF
# mimsave(output_file, images, loop=0, duration=1.0)

# In theory, 100Hz GIFs (duration=10) are supported by the GIF standard, but in practice, most players (especially browsers) cap out at 50Hz (duration=20). 
# This means that GIFs minimum delta is 10 ms in theory and 20 ms in practice.
duration = [1.0] * (final - initial + 1) + [2.0]
iio.imwrite(output_file, images, format="GIF", duration=duration, loop=0)

print("Simulation GIF created.")