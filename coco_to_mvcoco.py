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
    img_group = {k: g for k, g in itertools.groupby(imgs, lambda im: im["file_name"].rsplit(separator, 1)[0])}

    cats = [c["id"] for c in coco["categories"]]

    local_ann_id = 1
    for _, g in img_group.items():
        group_annotation_per_category = {cat: {} for cat in cats}
        for image in g:
            ext = image['file_name'].rsplit(separator, 1)[1]
            ext = ext.rsplit('.', 1)[0]
            image_annotations = [a for a in coco["annotations"] if a['image_id'] == image['id']]
            for cat in cats:
                group_annotation_per_category[cat][ext] = [a for a in image_annotations if a["category_id"] == cat]

        for c, groups in group_annotation_per_category.items():
            views = len(groups)
            lengths = sum([len(cat_anns) for _, cat_anns in groups.items()])

            if lengths != 0 and lengths != views:
                print(f"There is more than one object for images in category {c}")
            else:
                output_append = {
                    "id": local_ann_id,
                    "category_id": c
                }
                for v, annotations in groups.items():
                    del annotations["image_id"]
                    del annotations["category_id"]
                    output_append[v] = annotations[0]

                output["annotations"].append(output_append)
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
