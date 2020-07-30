import copy

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from yolov3.utils.boxes import rescale_boxes


def hsv2bgr(hsv):
    """
    Converts hsv to rgb
    :param hsv: h, s, v tuple
    :return: r, g, b tuple
    """
    h, s, v = hsv
    c = s * v
    x = c * (1 - abs(((h / 60) % 2) - 1))
    m = v - c
    if 0 <= h < 60:
        b, g, r = 0, x, c
    elif 60 <= h < 120:
        b, g, r = 0, c, x
    elif 120 <= h < 180:
        b, g, r = x, c, 0
    elif 180 <= h < 240:
        b, g, r = c, x, 0
    elif 240 <= h < 270:
        b, g, r = c, 0, x
    else:
        b, g, r = x, 0, c
    return int((b + m) * 255), int((g + m) * 255), int((r + m) * 255)


def draw_detections(image_path, out_path, detections, classes, model_img_size,
                    colors=None, gt=None, f=None, epipoints=None, with_name=True):
    """
    Draw detected bounding boxes in the image. If specified, it also draws ground truth boundinb boxes and epipolar
    lines
    :param image_path: path to the source image
    :param out_path: path of the output file
    :param detections: tensor with detections
    :param classes: list of class names
    :param model_img_size: image size fed to the model
    :param colors: list of colours
    :param gt: ground truth tensors
    :param f: fundamental matrix
    :param epipoints: points of the epipolar line
    :param with_name: boolean indicating if the name of the class must be added to the bounding box
    :return:
    """
    if colors is None:
        colors = ["red", "fuchsia", "lime", "black"]

    image = Image.open(image_path).convert("RGB")
    drawer = ImageDraw.Draw(image)
    im_w, im_h = image.size

    font = ImageFont.load("arial.pil")
    if gt is not None:
        g = copy.deepcopy(gt)
        g[:, 2:6] = rescale_boxes(g[:, 2:6], model_img_size, (im_h, im_w), normalized=True, xywh=True)
        for _, cls_idx, x1, y1, x2, y2 in g:
            cls = classes[int(cls_idx)]
            color = colors[int(cls_idx)]
            w = x2 - x1
            text_size = drawer.textsize(cls, font=font)
            x_centered = x1 - (text_size[0] - w) // 2
            drawer.rectangle([x1, y1, x2, y2], outline=color, width=3)
            if with_name:
                drawer.text((x_centered, y2), cls, fill=color, font=font)

    if detections is not None and len(detections) > 0:
        rescaled_detections = copy.deepcopy(detections)
        rescaled_detections = rescale_boxes(rescaled_detections, model_img_size, (im_h, im_w))
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in rescaled_detections:
            cls = classes[int(cls_pred)]
            color = colors[int(cls_pred)]
            w = x2 - x1
            text_size = drawer.textsize(cls, font=font)
            x_centered = x1 - (text_size[0] - w) // 2
            drawer.rectangle([x1, y1, x2, y2], outline=color, width=3)
            if with_name:
                drawer.text((x_centered, y1 - text_size[1]), cls, fill=color, font=font)

    if epipoints is not None:
        rescaled_epipoints = np.concatenate((epipoints[:, :2], np.zeros((len(epipoints), 2))), axis=1)
        rescaled_epipoints = rescale_boxes(rescaled_epipoints, model_img_size, (im_h, im_w),
                                           normalized=False, xywh=False)
        rescaled_epipoints = np.concatenate((rescaled_epipoints[:, :2], np.ones((len(epipoints), 1))), axis=1)
        for centre in rescaled_epipoints:
            epiline = f @ centre
            p0 = (0, - epiline[2] / epiline[1])
            p1 = (im_w, -(epiline[2] + epiline[0] * im_w) / epiline[1])
            drawer.line([p0, p1], fill="red", width=1)
    image.save(out_path, "PNG")
