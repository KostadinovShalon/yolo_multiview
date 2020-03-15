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
    def __init__(self, n_classes, views=("A", "B"), anchors=None, img_size=416, scales=3,
                 contrastive_factor=1, projected_location_factor=10):
        super(MVYOLOv3, self).__init__(n_classes, anchors, img_size, scales)
        projection_net_keys = permutations(views, r=2)
        self.permutation_indices = {perm: i for i, perm in enumerate(projection_net_keys)}
        self.views = views
        self.projection_network = nn.ModuleList()
        self.mv_yolo_layers = nn.ModuleList()
        self.img_size = img_size
        self.contrastive_factor = contrastive_factor
        self.projection_factor = projected_location_factor
        n_anchors = len(self.anchors)
        for i in range(scales):
            j = scales - i
            anchors_in_scale = self.anchors[(j - 1) * n_anchors // scales:j * n_anchors // scales]
            self.mv_yolo_layers.append(MVYOLOLayer(anchors_in_scale, n_classes, scales, views, img_size))

        for _ in self.permutation_indices:
            self.projection_network.append(nn.Sequential(
                    nn.Linear(4, 10),
                    nn.ReLU(),
                    nn.BatchNorm1d(10),
                    nn.Linear(10, 10),
                    nn.ReLU(),
                    nn.BatchNorm1d(10),
                    nn.Linear(10, 10),
                    nn.ReLU(),
                    nn.BatchNorm1d(10),
                    nn.Linear(10, 4),
                    nn.Sigmoid())
            )

    def get_projections(self, x, view_permutation):
        return self.projection_network[self.permutation_indices[view_permutation]](x)

    def forward(self, x, targets=None):
        def distance(f: torch.Tensor, h):
            return f.dist(h)
        output = {v: [] for v in self.views}
        yolo_outputs = []
        scale = 0
        early_xtra_ftrs = [None] * len(self.views)
        img_dim = self.img_size
        target_grid_vectors = {}
        for extra_features_subnet, yolo_conv_layer, mv_yolo_layer in \
                zip(self.extra_features, self.yolo_conv_layers, self.mv_yolo_layers):
            view_outputs = {}
            for i, view in enumerate(self.views):
                img = x[view]
                img_dim = img.shape[3]
                z = self.features_extractor(img)

                # Contrastive loss
                if targets and view not in target_grid_vectors.keys():
                    grids = z[-1].shape[-1]
                    view_target = targets[view]
                    v_x = view_target[:, 2]
                    v_y = view_target[:, 3]
                    v_w = view_target[:, 4]
                    v_h = view_target[:, 5]
                    v_cx = v_x + v_w/2
                    v_cy = v_y + v_h/2
                    v_ci = (v_cx * grids).floor().long()
                    v_cj = (v_cy * grids).floor().long()
                    target_grid_vectors[view] = []
                    for k, (ci, cj) in enumerate(zip(v_ci, v_cj)):
                        target_grid_vectors[view].append(z[-1][k, :, ci, cj])
                    target_grid_vectors[view] = torch.stack(target_grid_vectors[view], 0)

                xtra_ftrs = extra_features_subnet(z[-(scale + 1)], early_xtra_ftrs[i])
                y = xtra_ftrs[-1]
                early_xtra_ftrs[i] = xtra_ftrs[-2]

                outputs = yolo_conv_layer(y)
                view_outputs[view] = outputs
            yolo_outputs.append(view_outputs)
            scale += 1
        yolo_loss = 0
        projection_loss = 0
        contrastive_loss = 0
        # Projection output and loss
        for yolo_output, mv_yolo_layer in zip(yolo_outputs, self.mv_yolo_layers):

            pred_out = mv_yolo_layer(yolo_output, targets, img_dim)
            for v, out in pred_out.items():
                output[v].append(out["predictions"])

                if targets:
                    yolo_loss += out["sv_loss"]
                    matched_predictions = out["matched_predictions"]
                    if len(matched_predictions) > 1:
                        for other_view in self.views:
                            if v != other_view:
                                projected_predictions = self.get_projections(matched_predictions[:, :4], (v, other_view))
                                other_view_targets = targets[other_view]
                                projected_loss = nn.MSELoss()(projected_predictions[:, :4], other_view_targets[:, 2:6])

                                projection_loss += self.projection_factor * projected_loss

        if targets:
            # Constrastive Loss
            for i in range(len(self.views)):
                for j in range(i + 1, len(self.views)):
                    v = self.views[i]
                    u = self.views[j]
                    c_loss = distance(target_grid_vectors[v], target_grid_vectors[u])
                    contrastive_loss += self.contrastive_factor * c_loss

        yolo_outputs = {v: to_cpu(torch.cat(outs, 1))for v, outs in output.items()}

        return yolo_outputs if targets is None else ((yolo_loss, projection_loss, contrastive_loss), yolo_outputs)


class MVYOLOLayer(YOLOLayer):

    def __init__(self, anchors, num_classes, scales, views=("A", "B"), img_dim=416):
        super(MVYOLOLayer, self).__init__(anchors, num_classes, img_dim)
        self.scales = scales
        self.views = views

    def forward(self, x: Dict, targets: Dict = None, img_dim=None):
        self.img_dim = img_dim
        output = {}
        for view in self.views:
            base_predictions = x[view]
            num_samples = base_predictions.size(0)
            grid_size = base_predictions.size(2)

            prediction = (
                base_predictions
                    .view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
            )

            # Get outputs
            base_x = torch.sigmoid(prediction[..., 0])  # Center x
            base_y = torch.sigmoid(prediction[..., 1])  # Center y
            base_w = prediction[..., 2]  # Width
            base_h = prediction[..., 3]  # Height
            pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
            pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

            # If grid size does not match current we compute new offsets
            if grid_size != self.grid_size:
                # noinspection PyTypeChecker
                self.compute_grid_offsets(grid_size, cuda=base_predictions.is_cuda)

            # Add offset and scale with anchors
            pred_boxes = prediction[..., :4].clone().detach()
            pred_boxes[..., 0] = base_x.data + self.grid_x
            pred_boxes[..., 1] = base_y.data + self.grid_y
            pred_boxes[..., 2] = torch.exp(base_w.data) * self.anchor_w
            pred_boxes[..., 3] = torch.exp(base_h.data) * self.anchor_h

            base_output = torch.cat(
                (
                    pred_boxes.view(num_samples, -1, 4) * self.stride,
                    pred_conf.view(num_samples, -1, 1),
                    pred_cls.view(num_samples, -1, self.num_classes),
                ),
                -1,
            )
            if view not in output.keys():
                output[view] = {}
            output[view]["predictions"] = base_output
            if targets is not None:
                iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, pred_boxes_idx = \
                    build_targets(
                        pred_boxes=pred_boxes,
                        pred_cls=pred_cls,
                        target=targets[view],
                        anchors=self.scaled_anchors,
                        ignore_thres=self.ignore_thres,
                        include_tensor_idx=True
                    )
                obj_mask = obj_mask.type(torch.bool)
                noobj_mask = noobj_mask.type(torch.bool)
                # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
                base_loss_x = self.mse_loss(base_x[obj_mask], tx[obj_mask])
                base_loss_y = self.mse_loss(base_y[obj_mask], ty[obj_mask])
                base_loss_w = self.mse_loss(base_w[obj_mask], tw[obj_mask])
                base_loss_h = self.mse_loss(base_h[obj_mask], th[obj_mask])
                base_loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
                base_loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
                base_loss_conf = self.obj_scale * base_loss_conf_obj + self.noobj_scale * base_loss_conf_noobj
                base_loss_cls = self.ce_loss(pred_cls[obj_mask], tcls[obj_mask].argmax(1))
                base_total_loss = base_loss_x + base_loss_y + base_loss_w + base_loss_h + base_loss_conf + base_loss_cls

                output[view]["sv_loss"] = base_total_loss
                output[view]["matched_predictions"] = pred_boxes_idx
                # other_views = [v for v in self.views if v != view]

                # for other_view in other_views:
                #     other_view_projections = projections[(view, other_view)]
                #     num_samples = other_view_projections.size(0)
                #     grid_size = other_view_projections.size(2)
                #     # Projected bounding boxes must include local id
                #     projection = (
                #         other_view_projections.view(num_samples, self.num_anchors, 5, grid_size, grid_size)
                #             .permute(0, 1, 3, 4, 2)
                #             .contiguous()
                #     )
                #
                #     # Get outputs
                #     x = torch.sigmoid(projection[..., 0])  # Center x
                #     y = torch.sigmoid(projection[..., 1])  # Center y
                #     w = projection[..., 2]  # Width
                #     h = projection[..., 3]  # Height
                #
                #     # Add offset and scale with anchors
                #     proj_boxes = projection[..., :4].clone().detach()
                #     proj_boxes[..., 0] = x.data + self.grid_x
                #     proj_boxes[..., 1] = y.data + self.grid_y
                #     proj_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
                #     proj_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
                #
                #     other_view_targets = targets[other_view]
        return output
