import os
import xml.etree.ElementTree as ET

# Define your class mapping
class_map = {
    'PPE_Glass': 0,
    'PPE_Helmet': 1,
    'PPE_Gloves': 2,
    'PPE_Shoe': 3
}

# Folder containing XML files
xml_folder = r'copy your path\Desktop\PPE\Train\labels'  # <-- Update this path

# Output folder for YOLO TXT files
output_folder = os.path.join(xml_folder, 'yolo_labels')
os.makedirs(output_folder, exist_ok=True)

def convert(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    yolo_lines = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_map:
            continue

        class_id = class_map[class_name]
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    txt_filename = os.path.join(output_folder, os.path.splitext(os.path.basename(xml_file))[0] + '.txt')
    with open(txt_filename, 'w') as f:
        f.write('\n'.join(yolo_lines))

# Process all XML files
for file in os.listdir(xml_folder):
    if file.endswith('.xml'):
        convert(os.path.join(xml_folder, file))

print("✅ All XML files converted to YOLO format.")

