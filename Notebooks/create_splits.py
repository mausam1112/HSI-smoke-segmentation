import os
import glob
from collections import defaultdict

# Dataset directories
base_dir = './HSSDataset/HSS_VOC/'
image_dir = os.path.join(base_dir, 'npy_multichannel')
split_dir = os.path.join(base_dir, 'split')

# Create v1 and v2 dirs
for v in ['v1', 'v2']:
    for split in ['train', 'test', 'val']:
        path = os.path.join(split_dir, v, split + '.txt')
        os.makedirs(os.path.dirname(path), exist_ok=True)

# Find all .npy files
npy_files = glob.glob(os.path.join(image_dir, '*.npy'))

# Group filenames by scene (base name without .npy)
scene_to_files = defaultdict(list)
for file_path in npy_files:
    filename = os.path.basename(file_path)
    base_name = filename[:-4]  # remove .npy
    scene = filename.split('_')[0]
    scene_to_files[scene].append(base_name)

print("Scenes found:", sorted(scene_to_files.keys(), key=int))

# Define splits
splits = {
    'v1': {
        'train': list(range(1, 13)),
        'test': list(range(13, 19)),
        'val': [19, 20]
    },
    'v2': {
        'train': list(range(1, 7)) + list(range(13, 19)),
        'test': list(range(7, 13)),
        'val': [19, 20]
    }
}

# Write split files
for v, split_def in splits.items():
    for split_name, scene_list in split_def.items():
        filepath = os.path.join(split_dir, v, split_name + '.txt')
        with open(filepath, 'w') as f:
            for scene in sorted(scene_list):
                for base_name in sorted(scene_to_files[str(scene)]):
                    f.write(f"{base_name}\n")
        print(f"Wrote files for {v}/{split_name} from scenes {scene_list}")

print("Splits created successfully.")
