import os
import math

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import argparse
import json
from matplotlib import cm


def get_src_and_dst_points(coco, src_view, dst_view, add_padding=True, img_size=None, with_id=False, class_id=None):
    """
    Gets the source and destination points from the bounding boxes of a mv-coco annotation file
    :param coco: mv-coco annotation file
    :param src_view: the name of the source view
    :param dst_view: the name of the destination view
    :param add_padding: if square padding is needed. Default: True
    :param img_size: if different than None, this is the image size to rescale the points. Otherwise, it is in the
                    [-1, 1] range (as per COCO annotations)
    :param with_id: if img ids are to be returned
    :param class_id: if not None, return only src and dst points from the selected class id
    :return: a tuple of numpy arrays (of shape N x 2) with the src and dst points. If with_id is True, then an
                additional array with the img ids is also returned
    """
    dst = []
    src = []
    ids = []

    for ann in coco['annotations']:
        if class_id is not None:
            if ann["category_id"] != class_id:
                continue
        if src_view in ann["views"] and dst_view in ann["views"]:
            image_id = ann["image_group_id"]
            ids.append(image_id)
            image_group = next(im for im in coco["images"] if im["id"] == image_id)
            dst_image = image_group["views"][dst_view]
            src_image = image_group["views"][src_view]
            padding_dst = 0, 0
            padding_src = 0, 0
            dst_w, dst_h = dst_image["width"], dst_image["height"]
            src_w, src_h = src_image["width"], src_image["height"]
            dst_padded_w, dst_padded_h = dst_w, dst_h
            src_padded_w, src_padded_h = src_w, src_h
            if add_padding:
                pad = abs(dst_w - dst_h) / 2
                padding_dst = (0, pad) if dst_w > dst_h else (pad, 0)
                dst_padded_w, dst_padded_h = dst_padded_w + padding_dst[0] * 2, dst_padded_h + padding_dst[1] * 2
                pad = abs(src_w - src_h) / 2
                padding_src = (0, pad) if src_w > src_h else (pad, 0)
                src_padded_w, src_padded_h = src_padded_w + padding_src[0] * 2, src_padded_h + padding_src[1] * 2
            bbox_dst = ann["views"][dst_view]["bbox"]
            bbox_src = ann["views"][src_view]["bbox"]
            bbox_dst = (bbox_dst[0] + bbox_dst[2] / 2 + padding_dst[0], bbox_dst[1] + bbox_dst[3] / 2 + padding_dst[1])
            bbox_src = (bbox_src[0] + bbox_src[2] / 2 + padding_src[0], bbox_src[1] + bbox_src[3] / 2 + padding_src[1])
            if img_size:
                bbox_dst = bbox_dst[0] / dst_padded_w * img_size, bbox_dst[1] / dst_padded_h * img_size
                bbox_src = bbox_src[0] / src_padded_w * img_size, bbox_src[1] / src_padded_h * img_size
            dst.append(bbox_dst)
            src.append(bbox_src)
    src = np.array(src)
    dst = np.array(dst)
    if with_id:
        return src, dst, ids
    return src, dst


def get_eplilines(src_preds, f):
    if src_preds is None:
        return None
    src_centres = np.concatenate((src_preds[:, :2], np.ones((len(src_preds), 1))), axis=1)
    epilines = f @ src_centres.transpose()
    return epilines.transpose()


def compute_fundamental_matrix(coco, src_view, dst_view, img_size=None, class_id=None):

    src, dst = get_src_and_dst_points(coco, src_view, dst_view, add_padding=True, img_size=img_size, class_id=class_id)

    Ft, _ = ransac((src, dst), FundamentalMatrixTransform, min_samples=8,
                   residual_threshold=1, max_trials=5000)
    f = Ft.params
    error, std = compute_error_not_signed(coco, f, src_view, dst_view, img_size, class_id)
    return f, error, std


def compute_error(coco, fundamental_matrix, src_view, dst_view, img_size, class_id=None):
    src, dst = get_src_and_dst_points(coco, src_view, dst_view, img_size=img_size, class_id=class_id)
    distances = []
    for src_point, dst_point in zip(src, dst):
        epiline = epipolar_line(np.concatenate((src_point, [1])), fundamental_matrix)
        A, B, C = epiline[0], epiline[1], epiline[2]
        d = (A * dst_point[0] + B * dst_point[1] + C) / np.sqrt(A ** 2 + B ** 2)
        distances.append(d)

    return np.mean(distances), np.std(distances)


def compute_error_not_signed(coco, fundamental_matrix, src_view, dst_view, img_size, class_id=None):
    src, dst = get_src_and_dst_points(coco, src_view, dst_view, img_size=img_size, class_id=class_id)
    distances = []
    for src_point, dst_point in zip(src, dst):
        epiline = epipolar_line(np.concatenate((src_point, [1])), fundamental_matrix)
        A, B, C = epiline[0], epiline[1], epiline[2]
        d = np.abs(A * dst_point[0] + B * dst_point[1] + C) / np.sqrt(A ** 2 + B ** 2)
        distances.append(d)

    return 0, sum(xi ** 2 for xi in distances) / len(distances)


def epipolar_line(point, fundamental_matrix, im_width=None):
    epiline = fundamental_matrix @ point
    if im_width is None:
        return epiline
    else:
        ep0 = (0, - epiline[2] / epiline[1])
        ep1 = (im_width, -(epiline[2] + epiline[0] * im_width) / epiline[1])

        return epiline, ep0, ep1


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--annotations")
    args.add_argument("--vsrc", default="B")
    args.add_argument("--vdes", default="A")
    args.add_argument("--img_size", default=544, type=int)
    opts = args.parse_args()

    anns_file = opts.annotations
    src_view = opts.vsrc
    dst_view = opts.vdes
    img_size = opts.img_size

    with open(anns_file, 'r') as f:
        data = json.load(f)
    src_points, dst_points = get_src_and_dst_points(data, src_view, dst_view, add_padding=True, img_size=img_size)
    Ft, _ = ransac((src_points, dst_points), FundamentalMatrixTransform, min_samples=8,
                   residual_threshold=1, max_trials=5000)
    fundamental_matrix, error_m, error_std = compute_fundamental_matrix(data, src_view, dst_view)
    print(f"Fundamental matrix from view {src_view} to {dst_view}: {fundamental_matrix}")
    print(f"Distance error μ = {error_m}, σ = {error_std}")


if __name__ == '__main__':
    main()


def test_epipolar_lines(imgs_path, img_idx, coco_dict, src_view, dst_view, fundamental_matrix):
    #  Drawing epipolar lines
    annotations = coco_dict['annotations']
    images = coco_dict['images']
    test_ann = annotations[img_idx]
    bbox_dst = test_ann["views"][dst_view][0]["bbox"]

    img_file_name_dst = next(img["file_name"] for img in images
                             if img["id"] == test_ann["views"][dst_view][0]["image_id"])
    img_path = os.path.join(imgs_path, img_file_name_dst)

    im = Image.open(img_path)
    draw = ImageDraw.Draw(im)

    draw.rectangle([bbox_dst[0], bbox_dst[1], bbox_dst[0] + bbox_dst[2], bbox_dst[1] + bbox_dst[3]],
                   outline="red", width=3)
    draw.ellipse([bbox_dst[0] + bbox_dst[2] / 2 - 2, bbox_dst[1] + bbox_dst[3] / 2 - 2,
                  bbox_dst[0] + bbox_dst[2] / 2 + 2, bbox_dst[1] + bbox_dst[3] / 2 + 2],
                 fill="red")

    bbox_src = test_ann["views"][src_view][0]["bbox"]
    point_src = np.array([bbox_src[0] + bbox_src[2] / 2, bbox_src[1] + bbox_src[3] / 2, 1])
    _, ep0, ep1 = epipolar_line(point_src, fundamental_matrix, im.width)
    draw.line([ep0, ep1], fill='blue', width=1)

    im.show()


def test_epipolar_score(data_dict, idx, fundamental_matrix, src_view, dst_view):
    src_points, dst_points, ids = get_src_and_dst_points(data_dict, src_view, dst_view,
                                                         add_padding=True, with_id=True, img_size=544)
    src_points = np.concatenate((src_points, np.ones((src_points.shape[0], 1))), axis=1)
    epilines = fundamental_matrix @ src_points.transpose()
    epilines = epilines.transpose()
    distances = []
    high_error_images = []
    for i, (dst_point, epiline, _id) in enumerate(zip(dst_points, epilines, ids)):
        x, y = dst_point
        A, B, C = epiline
        d = np.abs(A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)
        if d > 20:
            im = next(im_g for im_g in data_dict["images"] if im_g["id"] == _id)
            high_error_images.append(im["tag"])
        distances.append(d)
        # print(d)
    error, std = np.mean(distances), np.std(distances)
    print(f"S error: {error} +- {std}")
    src_point = src_points[idx]
    epiline = fundamental_matrix @ src_point
    prob = np.ones((544, 544))
    delta = 0.99
    gamma = 0.67
    k = math.log(delta * (1 - gamma) / (gamma * (1 - delta))) / std
    x0 = std / (1 - math.log(gamma / (1 - gamma)) / math.log(delta / (1 - delta))) + error
    for i in range(544):
        for j in range(544):
            A, B, C = epiline
            d = np.abs(A * i + B * j + C) / np.sqrt(A ** 2 + B ** 2)
            p = 1 - 1 / (1 + math.exp(-k * (d - x0)))
            prob[j, i] = p
    img = Image.fromarray(np.uint8(cm.plasma(prob)[:, :, :3] * 255))
    img.save("test_ep.jpg")
    img.show()
