# adapted from https://github.com/bjornstenger/xml2yolo/blob/master/convert.py

import os
import glob
import shutil
from xml.dom import minidom

# Updated class mapping
class_mapping = {
    "car": 0,
    "truck": 1,
    "person": 2
}

def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def get_bounding_box_from_polygon(polygon):
    # Extract polygon coordinates (x1, y1, x2, y2, ..., x4, y4)
    x_coords = []
    y_coords = []
    for i in range(0, len(polygon), 2):  # Loop over the pairs of coordinates
        x_coords.append(float(polygon[i]))  # x1, x2, x3, x4
        y_coords.append(float(polygon[i+1]))  # y1, y2, y3, y4
    
    # Calculate bounding box from the polygon coordinates
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)
    
    return xmin, xmax, ymin, ymax

def convert_xml2yolo(source_dir, target_dir, class_mapping):
    # Search for XML files in all subdirectories using recursive glob pattern
    for fname in glob.glob(os.path.join('.', source_dir, '**', '*.xml'), recursive=True):
         # Parse XML file
        xmldoc = minidom.parse(fname)

        # Determine the output file path in the target directory
        fname_out = os.path.join(target_dir, os.path.relpath(fname, source_dir)[:-4] + '.txt')

        # Ensure target directory exists
        os.makedirs(os.path.dirname(fname_out), exist_ok=True)

        # Copy the corresponding image file (.jpg) to the target directory
        image_fname = fname[:-4] + '.jpg'  # Assuming the image has the same name as the XML
        if os.path.exists(image_fname):
            target_image_path = os.path.join(target_dir, os.path.relpath(image_fname, source_dir))
            os.makedirs(os.path.dirname(target_image_path), exist_ok=True)  # Ensure the target directory exists
            shutil.copy2(image_fname, target_image_path)
        
        # Open the YOLO label file for writing
        with open(fname_out, "w") as f:
            itemlist = xmldoc.getElementsByTagName('object')
            size = xmldoc.getElementsByTagName('size')[0]
            width = int((size.getElementsByTagName('width')[0]).firstChild.data)
            height = int((size.getElementsByTagName('height')[0]).firstChild.data)

            for item in itemlist:
                classid = (item.getElementsByTagName('name')[0]).firstChild.data
                if classid in class_mapping:
                    label_str = str(class_mapping[classid])
                else:
                    label_str = "-1"  # If the label isn't found in the mapping
                    print(f"Warning: label '{classid}' not in class_mapping")

                # skip point objects
                point = item.getElementsByTagName('point')
                if point:
                    print(f"Warning: Skipping object '{classid}' in {fname} because it is a point, not a bounding box.")
                    continue  # Skip this object

                # Check if bndbox exists and contains the necessary elements
                bndbox = item.getElementsByTagName('bndbox')
                if bndbox:
                    bndbox = bndbox[0]
                    xmin = float(bndbox.getElementsByTagName('xmin')[0].firstChild.data)
                    ymin = float(bndbox.getElementsByTagName('ymin')[0].firstChild.data)
                    xmax = float(bndbox.getElementsByTagName('xmax')[0].firstChild.data)
                    ymax = float(bndbox.getElementsByTagName('ymax')[0].firstChild.data)
                    
                    b = (xmin, xmax, ymin, ymax)
                    bb = convert_coordinates((width, height), b)
                    
                    f.write(label_str + " " + " ".join([f"{a:.6f}" for a in bb]) + '\n')
                else:
                    # Fallback to polygon if bndbox is missing
                    polygon = item.getElementsByTagName('polygon')
                    if polygon:
                        polygon = polygon[0]
                        coordinates = []
                        # Query the coordinates
                        for i in range(4):  # x1, x2, x3, x4, y1, y2, y3, y4
                            x = polygon.getElementsByTagName(f'x{i+1}')[0].firstChild.data
                            y = polygon.getElementsByTagName(f'y{i+1}')[0].firstChild.data
                            coordinates.append(x)
                            coordinates.append(y)

                        xmin, xmax, ymin, ymax = get_bounding_box_from_polygon(coordinates)
                        
                        b = (xmin, xmax, ymin, ymax)
                        bb = convert_coordinates((width, height), b)

                        f.write(label_str + " " + " ".join([f"{a:.6f}" for a in bb]) + '\n')
                    else:
                        print(f"Warning: Missing 'polygon' and 'bndbox' in {fname} for object '{classid}'")

        # print(f"Wrote {fname_out}")

def main():
    source_directory = "data/person"
    target_directory = "data/yolo-labelled/person"
    convert_xml2yolo(source_directory, target_directory, class_mapping)

if __name__ == '__main__':
    main()
