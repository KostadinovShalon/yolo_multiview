import torch


def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    :param boxes: tensor of normalized boxes
    :param current_dim: size of the squared image
    :param original_shape: height and width of the original image

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


# noinspection PyTypeChecker
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
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
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
    :param pred_boxes: predicted boxes, containing the x, y, w and h values in that order.
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

    byte_tensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    float_tensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    n_b = pred_boxes.size(0)
    n_a = pred_boxes.size(1)
    n_c = pred_cls.size(-1)
    n_g = pred_boxes.size(2)

    # Output tensors
    obj_mask = byte_tensor(n_b, n_a, n_g, n_g).fill_(0)
    noobj_mask = byte_tensor(n_b, n_a, n_g, n_g).fill_(1)
    class_mask = float_tensor(n_b, n_a, n_g, n_g).fill_(0)
    iou_scores = float_tensor(n_b, n_a, n_g, n_g).fill_(0)
    tx = float_tensor(n_b, n_a, n_g, n_g).fill_(0)
    ty = float_tensor(n_b, n_a, n_g, n_g).fill_(0)
    tw = float_tensor(n_b, n_a, n_g, n_g).fill_(0)
    th = float_tensor(n_b, n_a, n_g, n_g).fill_(0)
    tcls = float_tensor(n_b, n_a, n_g, n_g, n_c).fill_(0)

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
    :param prediction: tensor with the predictions. Last dimension must be a (5 + Classes)
    :param conf_thres: Confidence threshold for bounding boxes
    :param nms_thres: Non maximum supression threshold
    :return: detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output
