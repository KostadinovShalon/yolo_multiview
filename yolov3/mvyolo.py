import collections
from collections import OrderedDict
from itertools import permutations
from typing import Dict

import torch
import torch.nn as nn
import numpy as np

from yolov3.utils.boxes import build_targets
from yolov3.utils.networks import to_cpu
from yolov3.yolo import Darknet53, AnchorsConv, Upsample, YOLOLayer


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
        self.projection_conv_layers = {}
        self.projection_network = {}
        n_anchors = len(self.anchors)

        for permutation in projection_net_keys:
            self.projection_network[permutation] = nn.ModuleList()
            self.projection_conv_layers[permutation] = nn.ModuleList()

            for i in range(scales):
                self.projection_conv_layers[permutation].append(
                    nn.Conv2d(1024 // (2 ** i), n_anchors * 4 // scales, 1))
                if i == 0:
                    self.projection_network[permutation].append(AnchorsConv(1024, 1024))
                else:
                    self.projection_network[permutation].append(AnchorsConv(3 * 1024 // 2 ** (i + 1), 1024 // 2 ** i,
                                                                            pre=True))


class MVYOLOLayer(YOLOLayer):

    def __init__(self, anchors, num_classes, scales, views=("A", "B"), img_dim=416):
        super(MVYOLOLayer, self).__init__(anchors, num_classes, img_dim)
        self.scales = scales
        self.views = views

    def forward(self, x: Dict, targets: Dict = None, img_dim=None,
                projections: Dict = None, ftrs: Dict = None, *args, **kwargs):
        self.img_dim = img_dim
        output = {}
        for view in self.views:
            base_predictions = x[view]
            base_targets = targets[view]
            sv_output, sv_loss = super(MVYOLOLayer, self).forward(base_predictions, base_targets, img_dim)

            if view not in output.keys():
                output[view] = {}
            output[view]["predictions"] = sv_output

            other_views = [v for v in self.views if v != view]

            for other_view in other_views:
                other_view_projections = projections[(view, other_view)]
                num_samples = other_view_projections.size(0)
                grid_size = other_view_projections.size(2)
                # Projected bounding boxes must include local id
                projection = (
                    other_view_projections.view(num_samples, self.num_anchors, 5, grid_size, grid_size)
                        .permute(0, 1, 3, 4, 2)
                        .contiguous()
                )

                # Get outputs
                x = torch.sigmoid(projection[..., 0])  # Center x
                y = torch.sigmoid(projection[..., 1])  # Center y
                w = projection[..., 2]  # Width
                h = projection[..., 3]  # Height

                # Add offset and scale with anchors
                proj_boxes = projection[..., :4].clone().detach()
                proj_boxes[..., 0] = x.data + self.grid_x
                proj_boxes[..., 1] = y.data + self.grid_y
                proj_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
                proj_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

                other_view_targets = targets[other_view]

