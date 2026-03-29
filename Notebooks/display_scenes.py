import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Directory containing HSI .npy files
image_dir = './HSSDataset/HSS_VOC/npy_multichannel/'

# Find all .npy files
npy_files = glob.glob(os.path.join(image_dir, '*.npy'))

print(f"Found {len(npy_files)} HSI files.")

# Count files per scene
scene_counts = Counter()
for file_path in npy_files:
    filename = os.path.basename(file_path)
    scene = filename.split('_')[0]
    scene_counts[scene] += 1

# Group files by scene: take the first (lex smallest filename) per scene
scene_files = {}
for file_path in sorted(npy_files):
    filename = os.path.basename(file_path)
    scene = filename.split('_')[0]
    if scene not in scene_files:
        scene_files[scene] = file_path

scenes = sorted(scene_files.keys(), key=int)
num_scenes = len(scenes)
print(f"Unique scenes: {scenes}")
print(f"Displaying one image per scene (first frame):")

# Create subplots for images
cols = 5
rows = (num_scenes + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(20, 6.5 * rows))
if num_scenes == 0:
    plt.close()
else:
    axes = axes.flat if rows * cols > 1 else [axes]

for idx, scene in enumerate(scenes):
    ax = axes[idx]
    file_path = scene_files[scene]
    img = np.load(file_path).astype(np.float32)[:-1, :-1, :]
    vis_img = np.max(img, axis=-1)
    ax.imshow(vis_img, cmap='viridis')
    ax.set_title(f'Scene {scene}\\n{os.path.basename(file_path)}', fontsize=9, pad=10)
    ax.axis('off')

for idx in range(num_scenes, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, top=0.92)
plt.show()

# Bar graph: number of files per scene
fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
scenes_sorted = sorted(scene_counts.keys(), key=int)
counts = [scene_counts[s] for s in scenes_sorted]
ax_bar.bar(scenes_sorted, counts, color='skyblue', edgecolor='navy', alpha=0.7)
ax_bar.set_xlabel('Scene')
ax_bar.set_ylabel('Number of Files')
ax_bar.set_title('Number of HSI Files per Scene')
plt.setp(ax_bar.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("Images and bar graph displayed.")

print(f"1-6: {sum(counts[:6])}")
print(f"7-12: {sum(counts[6:12])}")
print(f"13-18: {sum(counts[12:18])}")
print(f"19-20: {sum(counts[18:20])}")
