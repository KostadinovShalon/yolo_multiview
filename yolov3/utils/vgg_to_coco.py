"""
Created on Thu Apr 25 2019
@author: Brian Isaac-Medina

@ref: coco_json_create_2.0.py by Neel
"""
import json
import os
from PIL import Image
from shapely import geometry
import argparse
from tqdm import tqdm


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


parser = argparse.ArgumentParser()
parser.add_argument('--imdirs', nargs='+', help="Image directories")
parser.add_argument('--vgg_files', nargs='+', help='vgg file paths, in the same order as imdirs')
parser.add_argument('--category_ids', type=int, nargs='+', help='Category ids, in the same order as imdirs')
parser.add_argument('--category_names', type=str, nargs='+', help='Category names, in the same order as imdirs')
parser.add_argument('--output', help="output_path", default='xray.json')

opts = parser.parse_args()

output_path = opts.output
im_dirs = opts.imdirs
vgg_files = opts.vgg_files
cat_ids = opts.category_ids
cat_names = opts.category_names

output = {
    "info": {
        "description": "X-Ray Image Dataset",
        "url": "https://www.durham.ac.uk",
        "version": "0.1",
        "year": 2020,
        "contributor": "ICG",
        "date_created": "25/04/2019"
    },
    "licenses": [{
        "url": "https://www.durham.ac.uk",
        "id": 0,
        "name": "Durham ICG, Research work"
    }],
    "images": [],
    "annotations": [],
    "categories": []
}

img_id = 1
annotation_id = 1
for im_dir, vgg_file, cat_id, cat_name in zip(im_dirs, vgg_files, cat_ids, cat_names):
    with open(vgg_file) as f:
        data = json.load(f)
    print(f'Handling category: {cat_name}')
    for data_id, data_info in tqdm(data.items()):
        img_name = data_info['filename']
        filename = find(img_name, im_dir)

        img = Image.open(filename)
        width, height = img.size
        img_info = {
            "license": 0,
            "file_name": img_name,
            "width": width,
            "height": height,
            "id": img_id
        }
        output["images"].append(img_info)

        regions = data_info['regions']
        polygons = []
        bbox = None
        area = 0
        for _, region in regions.items():
            ptx = region['shape_attributes']['all_points_x']
            pty = region['shape_attributes']['all_points_y']
            pts_zip = list(zip(ptx, pty))
            pts = [p for point in pts_zip for p in point]

            polygon = geometry.Polygon([[x, y] for x, y in pts_zip])
            x, y, max_x, max_y = polygon.bounds
            box_width = max_x - x
            box_height = max_y - y
            bbox = (x, y, box_width, box_height)
            area = area + polygon.area
            polygons.append(pts)

        output['annotations'].append({
            "segmentation": polygons,
            "iscrowd": 0,
            "image_id": img_id,
            "category_id": cat_id,
            "id": annotation_id,
            "bbox": bbox,
            "area": area
        })
        annotation_id = annotation_id + 1
        img_id = img_id + 1
    output["categories"].append({
        "supercategory": "xrayimage",
        "id": cat_id,
        "name": cat_name
    })

json_output = json.dumps(output)
with open(output_path, 'w') as output_file:
    output_file.write(json_output)
