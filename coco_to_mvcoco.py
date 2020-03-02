import json
import argparse
import copy
import itertools


def change_to_mv_coco(coco_file, separator):
    with open(coco_file, 'r') as f:
        coco = json.load(f)

    output = copy.deepcopy(coco)
    output["annotations"] = []

    imgs = sorted(coco["images"], key=lambda im: im["file_name"].lower())
    # img_group contains a dictionary of the form
    # {
    #     "common_name_of_image": [imgs]
    # }
    img_group = {k: list(g) for k, g in itertools.groupby(imgs, lambda im: im["file_name"].rsplit(separator, 1)[0])}

    cats = [c["id"] for c in coco["categories"]]  # List with category ids

    local_ann_id = 1
    for k, g in img_group.items():
        # The objective is to create a list of annotations of the following form
        # [{
        #     "id": local id of the annotation,
        #     "category_id": category id of the annotation
        #     "views": {
        #         "A": {
        #             "segmentation": [],
        #             "image_id": image id,
        #             "iscrowd": int,
        #             "bbox": [],
        #             "area": float
        #         },
        #         "B": { ... }, ...
        #     }
        # }, ...
        # ]
        group_annotation_per_category = {cat: {} for cat in cats}
        for image in g:
            suffix = image['file_name'].rsplit(separator, 1)[1]
            suffix, ext = suffix.rsplit('.', 1)
            # Getting annotations for the current image
            image_annotations = [a for a in coco["annotations"] if a['image_id'] == image['id']]
            for cat in cats:
                # group_annotations_per_category is a dictionary with the annotations of a group divided in categories
                #
                # {
                #     cat_id: {
                #         "A": [annotations],  # A and B are the different views
                #         "B": [annotations], ...
                #     }, ...
                # }
                group_annotation_per_category[cat][suffix] = [a for a in image_annotations if a["category_id"] == cat]

        for c, groups in group_annotation_per_category.items():
            views = len(groups)  # groups is a dictionary with the views. views is the number of views in the group
            lengths = sum([len(cat_anns) for _, cat_anns in groups.items()])  # number of annotations per cat

            if lengths != 0:
                if lengths == views:  # So, for this category, there's only one item in each view
                    valid_annotation = dict(id=local_ann_id, category_id=c, views={})
                    local_ann_id += 1
                    for v, annotations in groups.items():
                        valid_annotation["views"][v] = annotations
                    output["annotations"].append(valid_annotation)
                elif lengths % views == 0:  # In this case there's more than one object of the same cat across all views
                    n_objects = lengths // views
                    conflicts = {v: [a["id"] for a in annotations] for v, annotations in groups.items()}
                    print(f"There are {n_objects} objects in the group {k}. Annotations with conflicts: ")
                    for v, ann_ids in conflicts.items():
                        print(f"{v}: {ann_ids}")
                    print("For the moment, the fix has not been implemented. Try Later")
                else:
                    print("There's a problem with the annotation")

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
