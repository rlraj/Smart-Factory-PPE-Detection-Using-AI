import os
import matplotlib.pyplot as plt

class_names = ['glass', 'gloves', 'helmet', 'shoe']
class_counts = {i: 0 for i in range(len(class_names))}

label_folder = 'copy your path/Desktop/PPE/Train/labels'  # Update as needed

for filename in os.listdir(label_folder):
    if filename.endswith('.txt'):
        with open(os.path.join(label_folder, filename), 'r') as f:
            for line in f:
                parts = line.strip().split()
                try:
                    class_id = int(parts[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                except (ValueError, IndexError):
                    print(f"⚠️ Skipping invalid line in {filename}: {line.strip()}")

# Plotting
labels = [class_names[i] for i in class_counts]
counts = [class_counts[i] for i in class_counts]

plt.figure(figsize=(8, 6))
plt.bar(labels, counts, color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Class Distribution in YOLO Dataset')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


