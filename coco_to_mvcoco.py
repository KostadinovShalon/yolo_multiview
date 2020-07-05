import json
import argparse
import copy
import itertools
import numpy as np


def change_to_mv_coco(coco_file, separator):
    """
    Converts a annotation coco file to a multi-view coco-like annotation file
    :param coco_file: coco annotation file path
    :param separator: multi-view separator
    """
    with open(coco_file, 'r') as f:
        coco = json.load(f)

    output = copy.deepcopy(coco)
    output["images"] = []
    output["annotations"] = []
    cats_id = [c["id"] for c in coco["categories"]]  # List with category ids

    imgs = sorted(coco["images"], key=lambda im: im["file_name"].lower())
    img_group = {k: list(g) for k, g in itertools.groupby(imgs, lambda im: im["file_name"].rsplit(separator, 1)[0])}
    group_id = 1
    local_ann_id = 1
    for tag, imgs in img_group.items():
        group = {
            "id": group_id,
            "tag": tag,
            "license": 0,
            "views": {}
        }
        group_annotation_per_category = {cat: {} for cat in cats_id}
        for img in imgs:
            suffix = img['file_name'].rsplit(separator, 1)[1].rsplit('.', 1)[0]
            group["views"][suffix] = {
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"]
            }

            # Creating annotations
            """
            The objective is to create a list of annotations of the following form
            [{
                "id": local id of the annotation,
                "image_group_id": id of the image group
                "category_id": category id of the annotation
                "views": {
                    "A": {
                        "segmentation": [],
                        "iscrowd": int,
                        "bbox": [],
                        "area": float
                    },
                    "B": { ... }, ...
                }
            }, ...
            ]
            """
            # Getting annotations for the current image
            image_annotations = [a for a in coco["annotations"] if a['image_id'] == img['id']]
            for cat in cats_id:
                """
                group_annotations_per_category is a dictionary with the annotations of a group divided in categories

                {
                    cat_id: {
                        "A": [annotations],  # A and B are the different views
                        "B": [annotations], ...
                    }, ...
                }
                """
                group_annotation_per_category[cat][suffix] = [a for a in image_annotations if a["category_id"] == cat]

        output["images"].append(group)
        for c, groups in group_annotation_per_category.items():
            views = len(groups)  # groups is a dictionary with the views. views is the number of views in the group
            lengths = sum([len(cat_anns) for cat_anns in groups.values()])  # number of annotations per cat

            if lengths != 0:
                if lengths == views:  # So, for this category, there's only one item in each view
                    valid_annotation = dict(id=local_ann_id, image_group_id=group_id, category_id=c, views={})
                    local_ann_id += 1
                    for v, annotations in groups.items():
                        valid_annotation["views"][v] = {
                            "segmentation": annotations[0]["segmentation"],
                            "iscrowd": annotations[0]["iscrowd"],
                            "bbox": annotations[0]["bbox"],
                            "area": annotations[0]["area"]
                        }
                    output["annotations"].append(valid_annotation)
                elif lengths % views == 0:  # In this case there's more than one object of the same cat across all views
                    n_objects = lengths // views
                    conflicts = {v: [a["id"] for a in annotations] for v, annotations in groups.items()}
                    print(f"There are {n_objects} objects in the group {tag} of category {c}. Annotations with conflicts: ")
                    for v, ann_ids in conflicts.items():
                        print(f"{v}: {ann_ids}")
                    print("Solving by distance in the X coordinate")
                    valid_annotations = []
                    for _ in range(n_objects):
                        valid_annotations.append(dict(id=local_ann_id, image_group_id=group_id, category_id=c, views={}))
                        local_ann_id += 1
                    for v, annotations in groups.items():
                        x_coordinates = [a["bbox"][0] for a in annotations]
                        ordered_indices = np.argsort(x_coordinates)
                        for i, oi in enumerate(ordered_indices):
                            valid_annotations[i]["views"][v] = {
                                "segmentation": annotations[oi]["segmentation"],
                                "iscrowd": annotations[oi]["iscrowd"],
                                "bbox": annotations[oi]["bbox"],
                                "area": annotations[oi]["area"]
                            }
                else:
                    print("There's a problem with the annotation")
        group_id += 1

    return output


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--json_file", help="Path to json file")
    args.add_argument("--separator", default="_", help="Separator between images")
    args.add_argument("--output", help="Path to output file")
    opts = args.parse_args()

    output = change_to_mv_coco(opts.json_file, opts.separator)
    with open(opts.output, 'w') as f:
        json.dump(output, f)


if __name__ == '__main__':
    main()
