import copy
import scipy.special

import numpy as np
import torch


def rescale_boxes(boxes, current_dim, original_shape, normalized=False, xywh=False):
    """
    Rescales bounding boxes to the original shape
    :param boxes: tensor of normalized boxes
    :param current_dim: size of the squared image
    :param original_shape: height and width of the original image
    :param normalized: boolean indicating if data is normalized
    :param xywh: boolean indicating if boxes are in xywh format
    :return: tensor of rescaled boxes
    """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    if normalized:
        boxes = boxes * current_dim
    if xywh:
        boxes = xywh2xyxy(boxes)
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x, return_as_tuple=False):
    """
    Converts a set of x,y,w,h numbers to x1, y1, x2, y2
    :param x: coordinates in x,y,w,h
    :param return_as_tuple: indicates if the result should be returned as tuple instead of a vector
    :return: coordinates in x1,y1,x2,y2
    """
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2

    if return_as_tuple:
        return y[..., 0], y[..., 1], y[..., 2], y[..., 3]
    return y


def xyxy2xywh(x, return_as_tuple=False):
    """
    Converts a set of x,y,w,h numbers to x1, y1, x2, y2
    :param x: coordinates in x,y,w,h
    :param return_as_tuple: indicates if the result should be returned as tuple instead of a vector
    :return: coordinates in x1,y1,x2,y2
    """
    y = x.new(x.shape)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = torch.abs(x[..., 2] - x[..., 0])
    y[..., 3] = torch.abs(x[..., 3] - x[..., 1])

    if return_as_tuple:
        return y[..., 0], y[..., 1], y[..., 2], y[..., 3]
    return y


def bbox_wh_iou(wh1, wh2):
    """
    Gets the IOU from two concentric boxes
    :param wh1: width and height pair of the first box
    :param wh2: width and height pair of the second box
    :return: the IOU
    """
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area.item()
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    :param box1: coordinates of the bounding box
    :param box2: coordinates of the second bounding box
    :param x1y1x2y2: indicates if the coordinates are given in x1, y1, x2, y2 format. If False, it is assumed
        that they are in x, y, w, h form.

    :return: the IoU between the boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = xywh2xyxy(box1, return_as_tuple=True)
        b2_x1, b2_y1, b2_x2, b2_y2 = xywh2xyxy(box2, return_as_tuple=True)
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def build_targets(pred_boxes: torch.Tensor, pred_cls: torch.Tensor, target: torch.Tensor,
                  anchors: torch.Tensor, ignore_thres: float):
    """
    Gets the masks for the actual targets in the channel with the best suited anchor box. It also gets a mask
    where there is no predicted object. However, if a "no best anchor" channel has a bigger IOU than the
    specified anchor, it is also considered an object (thus, the no obj mask gets to 0 for that anchor channel).
    :param pred_boxes: predicted boxes, containing the center x, center y, w and h values in that order.
                       It must be a tensor with the following dimensions: samples x anchors x grid_w x grid_h x 4
    :param pred_cls: predicted classes tensor, with dimensions: samples x anchors x grid_w x grid_h x classes
    :param target: actual targets tensor, with dimensions: target_boxes x 6. The second dimensions of 6-dim vector
                    with the following values: (idx, label, X, Y, W, H)
    :param anchors: anchors tensor, with dimensions: anchors x 2. The second dimensions are the width and height.
    :param ignore_thres: threshold IOU value where a target box in any anchor channel (independently if it is not the
                            best anchor channel) is not considered a "no object"
    :return: a tuple with the following values:
            - IOU scores: tensor with dimensions samples x anchors x grid_w x grid_h with the IOU scores of the
                            predicted boxes only in the best anchor channel
            - Class mask: tensor with dimensions samples x anchors x grid_w x grid_h indicating if the class was
                            correctly detected, only in the best anchor channel
            - Obj mask: tensor with dimensions samples x anchors x grid_w x grid_h indicating the target
                        location in the grid, only in the best anchor channel
            - No obj mask: tensor with dimensions samples x anchors x grid_w x grid_h indicating where are
                        no objects in the grid.
            - Target x: tensor with dimensions samples x anchors x grid_w x grid_h with the target x value only
                        for those grid elements (and anchor channels) where the box was found.
            - Target y: tensor with dimensions samples x anchors x grid_w x grid_h with the target y value only
                        for those grid elements (and anchor channels) where the box was found.
            - Target w: tensor with dimensions samples x anchors x grid_w x grid_h with the target w value only
                        for those grid elements (and anchor channels) where the box was found.
            - Target h: tensor with dimensions samples x anchors x grid_w x grid_h with the target h value only
                        for those grid elements (and anchor channels) where the box was found.
            - Target confidence score: tensor with dimensions samples x anchors x grid_w x grid_h
                        with the target x value with the confidence score. It has the same value for the obj mask but
                        with float type.
    """

    n_b = pred_boxes.size(0)
    n_a = pred_boxes.size(1)
    n_c = pred_cls.size(-1)
    n_g = pred_boxes.size(2)

    device = pred_boxes.device

    # Output tensors
    obj_mask = torch.zeros((n_b, n_a, n_g, n_g), dtype=torch.uint8, device=device)
    noobj_mask = torch.ones((n_b, n_a, n_g, n_g), dtype=torch.uint8, device=device)
    class_mask = torch.zeros((n_b, n_a, n_g, n_g), device=device)
    iou_scores = torch.zeros((n_b, n_a, n_g, n_g), device=device)
    tx = torch.zeros((n_b, n_a, n_g, n_g), device=device)
    ty = torch.zeros((n_b, n_a, n_g, n_g), device=device)
    tw = torch.zeros((n_b, n_a, n_g, n_g), device=device)
    th = torch.zeros((n_b, n_a, n_g, n_g), device=device)
    tcls = torch.zeros((n_b, n_a, n_g, n_g, n_c), device=device)

    if target.shape[0] > 0:
        # Convert to position relative to box
        target_boxes = target[:, 2:6] * n_g
        gxy = target_boxes[:, :2]
        gwh = target_boxes[:, 2:]
        # Get anchors with best iou
        ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
        best_ious, best_n = ious.max(0)
        # Separate target values
        b, target_labels = target[:, :2].long().t()
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()
        # Set masks
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

        # Set noobj mask to zero where iou exceeds ignore threshold
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

        # Coordinates
        tx[b, best_n, gj, gi] = gx - gx.floor()
        ty[b, best_n, gj, gi] = gy - gy.floor()
        # Width and height
        tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
        th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1
        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
        iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    t_conf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, t_conf


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    :param prediction: predictions tensor. Last dimension must be a (5 + Classes)
    :param conf_thres: Confidence threshold for bounding boxes
    :param nms_thres: Non maximum supression threshold
    :return: detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = []
    for image_i, image_pred in enumerate(prediction):
        keep_boxes = nms(conf_thresholding(image_pred, conf_thres), nms_thres)
        output.append(torch.stack(keep_boxes) if keep_boxes else None)

    return output


def mv_filtering(mv_prediction, fundamental_matrices, conf_thres=0.5, nms_thres=0.4, p_value=0.95):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    :param fundamental_matrices: fundamental matrices dictionary of tuples
    :param mv_prediction: tensor with multi-view predictions. Last dimension must be a (5 + Classes)
    :param conf_thres: Confidence threshold for bounding boxes
    :param nms_thres: Non maximum supression threshold
    :param p_value: P-value to search bounding boxes near the epipolar line
    :return: detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    for k in mv_prediction.keys():
        mv_prediction[k][..., :4] = xywh2xyxy(mv_prediction[k][..., :4])
    views = mv_prediction.keys()
    reordered_preds = [dict(zip(mv_prediction, ps)) for ps in zip(*mv_prediction.values())]
    output = {v: [[] for _ in range(len(reordered_preds))] for v in views}
    for i, mv_preds in enumerate(reordered_preds):
        valid_pairs = []  # Contiene las predicciones validas por imagen
        for k, sv_preds in mv_preds.items():  # Procesando por vista
            weakly_filtered = conf_thresholding(sv_preds, conf_thres)  # tensor de N x 7
            if weakly_filtered is not None:
                detections_by_view = {k: weakly_filtered}
                for v in mv_preds.keys():
                    if v != k:
                        keys = list(fundamental_matrices.keys())
                        categories = list(set(tuple(zip(*keys))[-1]))
                        categories.sort()
                        f = [fundamental_matrices[(k, v, c)] for c in categories]
                        detections_by_view[v] = weak_detection(src_preds=weakly_filtered,
                                                               dst_preds=reordered_preds[i][v],
                                                               filter_th=conf_thres * (1 - p_value),
                                                               fundamental_matrices=f,
                                                               nms_th=nms_thres)  # Best matching box in dst view
                for j in range(weakly_filtered.shape[0]):
                    # is_valid = True
                    found = 0
                    for v in views:
                        if detections_by_view[v][j] is None:
                            found += 1
                    if found < len(views) - 1:
                        d = {v: detections_by_view[v][j] for v in views}
                        d["cat"] = int(weakly_filtered[j][-1])
                        d["combined_score"] = 1
                        for v in views:
                            if detections_by_view[v][j] is not None:
                                d["combined_score"] *= detections_by_view[v][j][4] * detections_by_view[v][j][5]
                            else:
                                d["combined_score"] *= 1 - found / (len(views) / 2)
                        valid_pairs.append(d)
        # Removing overlaps
        if len(valid_pairs) > 0:
            valid_pairs = {v: [valid_pair[v] for valid_pair in valid_pairs if valid_pair[v] is not None] for v in views}
            for v in views:
                pairs = valid_pairs[v]
                if len(pairs) > 0:
                    p = torch.stack(pairs)
                    score = p[:, 4] * p[:, 5]
                    p = p[(-score).argsort()]
                    output[v][i].extend(nms(p, nms_thres))

    output = {
        v: [torch.stack(view_outputs) if len(view_outputs) else None for view_outputs in output[v]] for v in views}
    return output


def conf_thresholding(raw_preds, conf_thres, true_cls=None):
    """
    Confidence thresholding of predictions
    :param raw_preds: N x (5 + classes) predictions tensor. Format: x, y, w, h, conf, class_1_conf, class_2_conf, ...
    :param conf_thres: confidence threshold
    :param true_cls: if the true class is given, the confidence for that class is used. Otherwise, the maximum class
    confidence is used
    :return: Thresholded N x 7 predictions tensor with the format x, y, w, h, conf, class_conf, class_id
    """
    # Filter out confidence scores below threshold
    image_pred = raw_preds[raw_preds[:, 4] >= conf_thres]
    # If none are remaining => process next image
    if not image_pred.size(0):
        return None
    # Object confidence times class confidence
    if true_cls is not None:
        score = image_pred[:, 4] * image_pred[:, 5 + true_cls]
    else:
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
    # Sort by it
    image_pred = image_pred[(-score).argsort()]
    if true_cls is not None:
        class_confs = image_pred[:, 5 + true_cls].unsqueeze(1)
        class_preds = torch.ones(class_confs.shape) * true_cls
    else:
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
    valid_preds = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
    return valid_preds


def score_thresholding(raw_preds, score_thres, aimed_class=None):
    """
    Score (confidence x class_confidence) thresholding of predictions
    :param raw_preds: N x (5 + classes) predictions tensor. Format: x, y, w, h, conf, class_1_conf, class_2_conf, ...
    :param score_thres: score threshold
    :param aimed_class: if the true class is given, the confidence for that class is used. If -1, all class_confidences
    are considered. If None, the maximum class confidence is used
    :return: Thresholded N x 7 predictions tensor with the format x, y, w, h, conf, class_conf, class_id
    """
    # Object confidence times class confidence
    if aimed_class == -1:
        classes = raw_preds.size(1) - 5
        image_pred = []
        n = raw_preds.size(0)
        for i in range(classes):
            image_pred.append(
                torch.cat((raw_preds[:, :5], raw_preds[:, 5 + i].unsqueeze(1), torch.ones((n, 1)) * i), dim=1))
        image_pred = torch.cat(image_pred)
        score = image_pred[:, 4] * image_pred[:, 5]
        image_pred = image_pred[(-score).argsort()]
    else:
        if aimed_class is None:
            score = raw_preds[:, 4] * raw_preds[:, 5:].max(1)[0]
        else:
            score = raw_preds[:, 4] * raw_preds[:, 5 + aimed_class]
        image_pred = raw_preds[(-score).argsort()]

    image_pred = image_pred[score.sort(descending=True)[0] >= score_thres]
    # If none are remaining => process next image
    if not image_pred.size(0):
        return None
    if aimed_class is None:
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
    elif aimed_class == -1:
        class_confs = image_pred[:, 5].unsqueeze(1)
        class_preds = image_pred[:, 6].unsqueeze(1)
    else:
        n = len(image_pred)
        class_confs = image_pred[:, 5 + aimed_class].unsqueeze(1)
        class_preds = torch.ones([n, 1]) * aimed_class
    return torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)


def nms(dets, nms_thres):
    """
    Non maximum supression
    :param dets: N x 7 predictions tensor with the format: x1, y1, x2, y2, conf, class_conf, class_id
    :param nms_thres: NMS threshold
    :return: list of combined tensors after NMS or None if dets is None
    """
    if dets is None:
        return None
    keep_boxes = []
    while dets.size(0):
        large_overlap = bbox_iou(dets[0, :4].unsqueeze(0), dets[:, :4]) > nms_thres
        label_match = dets[0, -1] == dets[:, -1]
        # Indices of boxes with lower confidence scores, large IOUs and matching labels
        invalid = large_overlap & label_match
        weights = dets[invalid, 4:5]
        # Merge overlapping bboxes by order of confidence
        if weights.size(0) != 1:
            dets[0, :4] = (weights * dets[invalid, :4]).sum(0) / weights.sum()
        keep_boxes += [dets[0]]
        dets = dets[~invalid]
    return keep_boxes


def weak_detection(src_preds, dst_preds, filter_th, fundamental_matrices, nms_th):
    """
    Filters bounding boxes by confidence threshold and epipolar constraints given some src detections in one view
    and raw detections in the desired filtered view.
    :param src_preds: list with 7-dimensional tensors with the format: [x1, y1, x2, y2, conf, class_conf, class]
    :param dst_preds: raw Nx(5 + classes) tensor with the format [x1, y1, x2, y2, conf, class1_conf, class_cof, ...]
    :param filter_th: threshold probability for epipolar filtering
    :param fundamental_matrices: a dictionary of fundamental matrix triplets (fundamental_matrix, mean, std) where
    the key value is the source class.
    :param nms_th: non-maximum supression threshold
    :return: list with the same size as src_preds with the filtered tensors of bounding boxes. If no bound box was
    found in the destination view, a None value is assigned.
    """
    # Filter out confidence scores below threshold
    src_centres = copy.deepcopy(src_preds)
    src_centres[..., :4] = xyxy2xywh(src_centres[..., :4])
    src_centres = np.concatenate((src_centres[:, :2], np.ones((len(src_centres), 1))), axis=1)
    # epilines = fundamental_matrix @ src_centres.transpose()

    weaks = []

    for i, (src_centre, src_pred) in enumerate(zip(src_centres, src_preds)):
        src_class = src_pred[-1].cpu().long().item()
        dst_dets = conf_thresholding(dst_preds, 1e-2, src_class)
        if dst_dets is None:
            weaks.append(None)
            continue
        f, error, std = fundamental_matrices[src_class]
        epiline = f @ src_centre.transpose()
        A, B, C = epiline[0], epiline[1], epiline[2]
        x = (dst_dets[:, 0] + dst_dets[:, 2]) / 2
        y = (dst_dets[:, 1] + dst_dets[:, 3]) / 2
        d = (A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)
        distance_probs = prob_by_distance(d, error, std)
        dst_dets[:, 4] = dst_dets[:, 4] * distance_probs
        score_with_distance = dst_dets[:, 4] * dst_dets[:, 5]
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        valid_weak_preds = dst_dets[score_with_distance > filter_th]
        # If none are remaining => process next image
        if not valid_weak_preds.size(0):
            weaks.append(None)
            continue
        # Object confidence times class confidence
        # Sort by it
        score_with_distance = valid_weak_preds[:, 4] * valid_weak_preds[:, 5]
        valid_weak_preds = valid_weak_preds[(-score_with_distance).argsort()]
        valid_weak_preds = torch.stack(nms(valid_weak_preds, nms_th))
        weaks.append(valid_weak_preds[0])
    return weaks


def prob_by_distance(distance, error, std):
    """
    Epipolar probability
    :param distance: distance from the point to the epipolar line
    :param error: mean of the error
    :param std: standard deviation of the error
    :return: the value of 1 - erf(| distance - error | / (sqrt(2) * std) )
    """
    return scipy.special.erfc(np.abs(distance - error) / (np.sqrt(2) * std))
