import copy

import numpy as np
from PIL import Image, ImageDraw

from yolov3.utils.boxes import rescale_boxes


def plot_lines(x, y, vis, opts=None, env='main', win=None):
    if isinstance(y, dict):
        series = np.column_stack(list(y.values()))
        legends = list(y.keys())
    else:
        series = y
        legends = None

    if opts is None:
        opts = dict()
    opts['legend'] = legends
    plot_vals = dict(
        X=x,
        Y=series,
        opts=opts,
        env=env
    )

    if win is not None:
        plot_vals['win'] = win
    return vis.line(**plot_vals)


def hsv2bgr(hsv):
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


def draw_detections(image_path, out_path, detections, classes, model_img_size, colors=None, gt=None, f = None, epipoints=None):
    if colors is None:
        colors = [hsv2bgr((h, 1, 1)) for h in range(120, 240, 120 // len(classes))]

    image = Image.open(image_path).convert("RGB")
    drawer = ImageDraw.Draw(image)
    im_w, im_h = image.size

    if gt is not None:
        g = copy.deepcopy(gt)
        g[:, 2:6] = rescale_boxes(g[:, 2:6], model_img_size, (im_h, im_w), normalized=True, xywh=True)
        for _, cls_idx, x1, y1, x2, y2 in g:
            cls = classes[int(cls_idx)]
            w = x2 - x1
            text_size = drawer.textsize(cls)
            x_centered = x1 - (text_size[0] - w) // 2
            drawer.rectangle([x1, y1, x2, y2], outline="black", width=1)
            drawer.text((x_centered, y2), cls, fill="black")

    # Detections adjustment
    if detections is not None and len(detections) > 0:
        rescaled_detections = copy.deepcopy(detections)
        rescaled_detections = rescale_boxes(rescaled_detections, model_img_size, (im_h, im_w))
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in rescaled_detections:
            cls = classes[int(cls_pred)]
            color = colors[int(cls_pred)]
            w = x2 - x1
            text_size = drawer.textsize(cls)
            x_centered = x1 - (text_size[0] - w) // 2
            drawer.rectangle([x1, y1, x2, y2], outline=color, width=2)
            drawer.text((x_centered, y1 - text_size[1]), cls, fill=color)

    if epipoints is not None:
        rescaled_epipoints = np.concatenate((epipoints[:, :2], np.zeros((len(epipoints), 2))), axis=1)
        rescaled_epipoints = rescale_boxes(rescaled_epipoints, model_img_size, (im_h, im_w), normalized=False, xywh=False)
        rescaled_epipoints = np.concatenate((rescaled_epipoints[:, :2], np.ones((len(epipoints), 1))), axis=1)
        # rescaled_epipoints = xyxy2xywh(torch.tensor(rescaled_epipoints))
        for centre in rescaled_epipoints:
            epiline = f @ centre
            p0 = (0, - epiline[2] / epiline[1])
            p1 = (im_w, -(epiline[2] + epiline[0] * im_w) / epiline[1])
            drawer.line([p0, p1], fill='blue', width=1)
    image.save(out_path, "PNG")


def draw_boxes(image_path, cls_id, new_ann, samet_anns):
    image = Image.open(image_path).convert("RGB")
    drawer = ImageDraw.Draw(image)
    x1, y1, w, h = new_ann
    x2 = x1 + w
    y2 = y1 + h
    text_size = drawer.textsize(f"NEW [{cls_id}]")
    x_centered = x1 - (text_size[0] - w) // 2
    drawer.rectangle([x1, y1, x2, y2], outline="red", width=2)
    drawer.text((x_centered, y2), f"NEW [{cls_id}]", fill="red")

    for sann in samet_anns:
        x1, y1, w, h = sann
        x2 = x1 + w
        y2 = y1 + h
        text_size = drawer.textsize(f"SAMMET [{cls_id}]")
        x_centered = x1 - (text_size[0] - w) // 2
        drawer.rectangle([x1, y1, x2, y2], outline="black", width=2)
        drawer.text((x_centered, y1 - text_size[1]), f"SAMMET [{cls_id}]", fill="black")
    return image


def draw_dets(image_path, im_anns, cats, output_path):
    image = Image.open(image_path).convert("RGB")
    drawer = ImageDraw.Draw(image)
    for ann in im_anns:
        cat_id = ann['category_id']
        cat_name = next(c for c in cats if c["id"] == cat_id)
        cat_name = cat_name["name"]
        x1, y1, w, h = ann['bbox']
        x2 = x1 + w
        y2 = y1 + h
        text_size = drawer.textsize(cat_name)
        x_centered = x1 - (text_size[0] - w) // 2
        drawer.rectangle([x1, y1, x2, y2], outline="red", width=3)
        drawer.text((x_centered, y1 - text_size[1]), cat_name, fill="red")
    image.save(output_path)

