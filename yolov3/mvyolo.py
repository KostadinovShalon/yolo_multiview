import collections
from collections import OrderedDict
from itertools import  permutations

import torch
import torch.nn as nn
import numpy as np

from yolov3.utils.boxes import build_targets
from yolov3.utils.networks import to_cpu
from yolov3.yolo import Darknet53, AnchorsConv, Upsample


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = collections.OrderedDict()
        self.img_dim = img_dim
        self.grid_size = 0  # grid size
        self.stride = self.grid_x = self.grid_y = self.scaled_anchors = self.anchor_w = self.anchor_h = None

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            # noinspection PyTypeChecker
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = prediction[..., :4].clone().detach()
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
            obj_mask = obj_mask.type(torch.bool)
            noobj_mask = noobj_mask.type(torch.bool)
            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.ce_loss(pred_cls[obj_mask], tcls[obj_mask].argmax(1))
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics["grid_size"] = grid_size
            self.metrics["loss"] = to_cpu(total_loss).item()
            self.metrics["x"] = to_cpu(loss_x).item()
            self.metrics["y"] = to_cpu(loss_y).item()
            self.metrics["w"] = to_cpu(loss_w).item()
            self.metrics["h"] = to_cpu(loss_h).item()
            self.metrics["conf"] = to_cpu(loss_conf).item()
            self.metrics["cls"] = to_cpu(loss_cls).item()
            self.metrics["cls_acc"] = to_cpu(cls_acc).item()
            self.metrics["recall50"] = to_cpu(recall50).item()
            self.metrics["recall75"] = to_cpu(recall75).item()
            self.metrics["precision"] = to_cpu(precision).item()
            self.metrics["conf_obj"] = to_cpu(conf_obj).item()
            self.metrics["conf_noobj"] = to_cpu(conf_noobj).item()

            return output, total_loss


# noinspection PyTypeChecker
class _YOLOv3(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, n_classes, anchors=None, img_size=416, scales=3):
        super(_YOLOv3, self).__init__()
        self.pretrained_last_layer_wts = [None, None, None]
        self.pretrained_last_layer_bias = [None, None, None]

        self.anchors = anchors
        if anchors is None:
            self.anchors = [(10, 13), (16, 30), (33, 23),
                            (30, 61), (62, 45), (59, 119),
                            (116, 90), (156, 198), (373, 326)]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

        self.features_extractor = Darknet53()
        self.extra_features = nn.ModuleList()
        self.yolo_conv_layers = nn.ModuleList()
        self.yolo_layers = nn.ModuleList()
        self.scales = scales
        n_anchors = len(self.anchors)
        for i in range(scales):
            j = scales - i
            anchors_in_scale = self.anchors[(j - 1) * n_anchors // scales:j * n_anchors // scales]
            self.yolo_conv_layers.append(nn.Conv2d(1024 // (2 ** i), n_anchors * (n_classes + 5) // scales, 1))
            self.yolo_layers.append(YOLOLayer(anchors_in_scale, n_classes, img_size))
            if i == 0:
                self.extra_features.append(AnchorsConv(1024, 1024))
            else:
                self.extra_features.append(AnchorsConv(3 * 1024 // 2 ** (i + 1), 1024 // 2 ** i, pre=True))

    def forward(self, x, targets=None):
        # x is a BxCxWxH tensor and targets is a T'x6 tensor, where T' is the number of targets in the whole batch
        # The 6 dimensions of the second tensor dimensions of targets is batch_index, class, x, y, w, h
        img_dim = x.shape[3]
        loss = 0

        z = self.features_extractor(x)
        scale = 1
        early_xtra_ftrs = None
        yolo_outputs = []
        for extra_features_subnet, yolo_conv_layer, yolo_layer in \
                zip(self.extra_features, self.yolo_conv_layers, self.yolo_layers):
            xtra_ftrs = extra_features_subnet(z[-scale], early_xtra_ftrs)
            y = xtra_ftrs[-1]
            early_xtra_ftrs = xtra_ftrs[-2]

            outputs = yolo_conv_layer(y)
            outputs, layer_loss = yolo_layer(outputs, targets, img_dim)
            loss += layer_loss
            yolo_outputs.append(outputs)

            scale += 1
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))

        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_yolov3_weights(self, weights_path):
        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Loading weights of Darknet53
        ptr = 0
        ptr = self.load_conv_layer(self.features_extractor.c1[0], ptr, weights, self.features_extractor.c1[1])
        ptr = self.load_conv_layer(self.features_extractor.c2[0], ptr, weights, self.features_extractor.c2[1])
        for residual_block in self.features_extractor.r1:
            for i in range(0, len(residual_block), 3):
                ptr = self.load_conv_layer(residual_block[i], ptr, weights, residual_block[i + 1])
        ptr = self.load_conv_layer(self.features_extractor.c3[0], ptr, weights, self.features_extractor.c3[1])
        for residual_block in self.features_extractor.r2:
            for i in range(0, len(residual_block), 3):
                ptr = self.load_conv_layer(residual_block[i], ptr, weights, residual_block[i + 1])
        ptr = self.load_conv_layer(self.features_extractor.c4[0], ptr, weights, self.features_extractor.c4[1])
        for residual_block in self.features_extractor.r3:
            for i in range(0, len(residual_block), 3):
                ptr = self.load_conv_layer(residual_block[i], ptr, weights, residual_block[i + 1])
        ptr = self.load_conv_layer(self.features_extractor.c5[0], ptr, weights, self.features_extractor.c5[1])
        for residual_block in self.features_extractor.r4:
            for i in range(0, len(residual_block), 3):
                ptr = self.load_conv_layer(residual_block[i], ptr, weights, residual_block[i + 1])
        ptr = self.load_conv_layer(self.features_extractor.c6[0], ptr, weights, self.features_extractor.c6[1])
        for residual_block in self.features_extractor.r5:
            for i in range(0, len(residual_block), 3):
                ptr = self.load_conv_layer(residual_block[i], ptr, weights, residual_block[i + 1])

        current_ptr = ptr
        # Extra-ftrs layer
        for extra_feature_layer in self.extra_features:
            if extra_feature_layer.with_pre:
                ptr = self.load_conv_layer(extra_feature_layer.pre[0], ptr, weights, extra_feature_layer.pre[1])
            ptr = self.load_conv_layer(extra_feature_layer.c1[0], current_ptr, weights, extra_feature_layer.c1[1])
            ptr = self.load_conv_layer(extra_feature_layer.c2[0], ptr, weights, extra_feature_layer.c2[1])
            ptr = self.load_conv_layer(extra_feature_layer.c3[0], ptr, weights, extra_feature_layer.c3[1])
            ptr = self.load_conv_layer(extra_feature_layer.c4[0], ptr, weights, extra_feature_layer.c4[1])
            ptr = self.load_conv_layer(extra_feature_layer.c5[0], ptr, weights, extra_feature_layer.c5[1])
            ptr = self.load_conv_layer(extra_feature_layer.c6[0], ptr, weights, extra_feature_layer.c6[1])

            ptr += self.yolo_conv_layers[0].bias.numel()

    @staticmethod
    def load_conv_layer(conv_layer, ptr, weights, bn_layer=None):
        if bn_layer:
            bn_terms = [bn_layer.bias, bn_layer.weight, bn_layer.running_mean, bn_layer.running_var]
            num_b = bn_layer.bias.numel()  # Number of biases

            for bn_term in bn_terms:
                param = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_term)
                bn_term.data.copy_(param)
                ptr += num_b
        else:
            # Load conv. bias
            num_b = conv_layer.bias.numel()
            conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
            conv_layer.bias.data.copy_(conv_b)
            ptr += num_b
        # Load conv. weights
        num_w = conv_layer.weight.numel()
        conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
        conv_layer.weight.data.copy_(conv_w)
        ptr += num_w
        return ptr

    def get_metrics(self):
        metric_labels = [
            "grid_size",
            "loss",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "cls",
            "cls_acc",
            "recall50",
            "recall75",
            "precision",
            "conf_obj",
            "conf_noobj",
        ]
        metrics = []
        for i, metric in enumerate(metric_labels):
            formats = {m: "%.6f" for m in metric_labels}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in self.yolo_layers]
            metrics += [[metric, *row_metrics]]
        return metrics


class MVYOLOv3(_YOLOv3):
    def __init__(self, n_classes, views=("A", "B"), anchors=None, img_size=416, scales=3):
        super(MVYOLOv3, self).__init__(n_classes, anchors, img_size, scales)
        projection_net_keys = permutations(views, r=2)
        self.projection_conv_layers = nn.ModuleList()
        self.projection_network = nn.ModuleList()
        for i in range(scales):
            j = scales - i
            anchors_in_scale = self.anchors[(j - 1) * n_anchors // scales:j * n_anchors // scales]
            self.yolo_conv_layers.append(nn.Conv2d(1024 // (2 ** i), n_anchors * (n_classes + 5) // scales, 1))
            self.yolo_layers.append(YOLOLayer(anchors_in_scale, n_classes, img_size))
            if i == 0:
                self.extra_features.append(AnchorsConv(1024, 1024))
            else:
                self.extra_features.append(AnchorsConv(3 * 1024 // 2 ** (i + 1), 1024 // 2 ** i, pre=True))
