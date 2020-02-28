import collections
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from yolov3.utils.boxes import build_targets
from yolov3.utils.networks import to_cpu


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, _):
        pass


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
class Darknet53(nn.Module):

    def __init__(self):
        super(Darknet53, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            (f"conv_0", nn.Conv2d(3, 32, 3, padding=1, bias=False)),
            (f"batch_norm_0", nn.BatchNorm2d(32, momentum=0.9, eps=1e-5)),
            (f"leaky_0", nn.LeakyReLU(0.1))
        ]))

        self.c2 = nn.Sequential(OrderedDict([
            (f"conv_1", nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False)),
            (f"batch_norm_1", nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)),
            (f"leaky_1", nn.LeakyReLU(0.1))
        ]))

        self.r1 = nn.ModuleList()
        self.r1.append(nn.Sequential(OrderedDict([
            (f"conv_2", nn.Conv2d(64, 32, 1, bias=False)),
            (f"batch_norm_2", nn.BatchNorm2d(32, momentum=0.9, eps=1e-5)),
            (f"leaky_2", nn.LeakyReLU(0.1)),
            (f"conv_3", nn.Conv2d(32, 64, 3, padding=1, bias=False)),
            (f"batch_norm_3", nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)),
            (f"leaky_3", nn.LeakyReLU(0.1))
        ])))

        self.c3 = nn.Sequential(OrderedDict([
            (f"conv_4", nn.Conv2d(64, 128, 3, padding=1, stride=2, bias=False)),
            (f"batch_norm_4", nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)),
            (f"leaky_4", nn.LeakyReLU(0.1))
        ]))

        self.r2 = nn.ModuleList()
        for i in range(2):
            self.r2.append(nn.Sequential(OrderedDict([
                (f"conv_{5 + 2 * i}", nn.Conv2d(128, 64, 1, bias=False)),
                (f"batch_norm_{5 + 2 * i}", nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)),
                (f"leaky_{5 + 2 * i}", nn.LeakyReLU(0.1)),
                (f"conv_{5 + 2 * i + 1}", nn.Conv2d(64, 128, 3, padding=1, bias=False)),
                (f"batch_norm_{5 + 2 * i + 1}", nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)),
                (f"leaky_{5 + 2 * i + 1}", nn.LeakyReLU(0.1))
            ])))

        self.c4 = nn.Sequential(OrderedDict([
            (f"conv_9", nn.Conv2d(128, 256, 3, padding=1, stride=2, bias=False)),
            (f"batch_norm_9", nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)),
            (f"leaky_9", nn.LeakyReLU(0.1))
        ]))

        self.r3 = nn.ModuleList()
        for i in range(8):
            self.r3.append(nn.Sequential(OrderedDict([
                (f"conv_{10 + 2 * i}", nn.Conv2d(256, 128, 1, bias=False)),
                (f"batch_norm_{10 + 2 * i}", nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)),
                (f"leaky_{10 + 2 * i}", nn.LeakyReLU(0.1)),
                (f"conv_{10 + 2 * i + 1}", nn.Conv2d(128, 256, 3, padding=1, bias=False)),
                (f"batch_norm_{10 + 2 * i + 1}", nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)),
                (f"leaky_{10 + 2 * i + 1}", nn.LeakyReLU(0.1))
            ])))

        self.c5 = nn.Sequential(OrderedDict([
            (f"conv_26", nn.Conv2d(256, 512, 3, padding=1, stride=2, bias=False)),
            (f"batch_norm_26", nn.BatchNorm2d(512, momentum=0.9, eps=1e-5)),
            (f"leaky_26", nn.LeakyReLU(0.1))
        ]))

        self.r4 = nn.ModuleList()
        for i in range(8):
            self.r4.append(nn.Sequential(OrderedDict([
                (f"conv_{27 + 2 * i}", nn.Conv2d(512, 256, 1, bias=False)),
                (f"batch_norm_{27 + 2 * i}", nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)),
                (f"leaky_{27 + 2 * i}", nn.LeakyReLU(0.1)),
                (f"conv_{27 + 2 * i + 1}", nn.Conv2d(256, 512, 3, padding=1, bias=False)),
                (f"batch_norm_{27 + 2 * i + 1}", nn.BatchNorm2d(512, momentum=0.9, eps=1e-5)),
                (f"leaky_{27 + 2 * i + 1}", nn.LeakyReLU(0.1))
            ])))

        self.c6 = nn.Sequential(OrderedDict([
            (f"conv_43", nn.Conv2d(512, 1024, 3, padding=1, stride=2, bias=False)),
            (f"batch_norm_43", nn.BatchNorm2d(1024, momentum=0.9, eps=1e-5)),
            (f"leaky_43", nn.LeakyReLU(0.1))
        ]))

        self.r5 = nn.ModuleList()
        for i in range(4):
            self.r5.append(nn.Sequential(OrderedDict([
                (f"conv_{44 + 2 * i}", nn.Conv2d(1024, 512, 1, bias=False)),
                (f"batch_norm_{44 + 2 * i}", nn.BatchNorm2d(512, momentum=0.9, eps=1e-5)),
                (f"leaky_{44 + 2 * i}", nn.LeakyReLU(0.1)),
                (f"conv_{44 + 2 * i + 1}", nn.Conv2d(512, 1024, 3, padding=1, bias=False)),
                (f"batch_norm_{44 + 2 * i + 1}", nn.BatchNorm2d(1024, momentum=0.9, eps=1e-5)),
                (f"leaky_{44 + 2 * i + 1}", nn.LeakyReLU(0.1))
            ])))

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)

        for module in self.r1:
            x = x + module(x)

        x = self.c3(x)

        for module in self.r2:
            x = x + module(x)

        x_r3 = self.c4(x)

        for module in self.r3:
            x_r3 = x_r3 + module(x_r3)

        x_r4 = self.c5(x_r3)

        for module in self.r4:
            x_r4 = x_r4 + module(x_r4)

        x = self.c6(x_r4)

        for module in self.r5:
            x = x + module(x)

        return x_r3, x_r4, x


class AnchorsConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AnchorsConv, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1, bias=False),
            nn.BatchNorm2d(out_channels // 2, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, 1, bias=False),
            nn.BatchNorm2d(out_channels // 2, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1)
        )

        self.c5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, 1, bias=False),
            nn.BatchNorm2d(out_channels // 2, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1)
        )

        self.c6 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x_p = self.c5(x)
        x = self.c6(x_p)

        return x_p, x

    def apply(self, fn):
        fn(self.c1)
        fn(self.c2)
        fn(self.c3)
        fn(self.c4)
        fn(self.c5)
        fn(self.c6)


# noinspection PyTypeChecker
class YOLOv3(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, n_classes, anchors=None, img_size=416):
        super(YOLOv3, self).__init__()
        self.pretrained_last_layer_wts = [None, None, None]
        self.pretrained_last_layer_bias = [None, None, None]

        if anchors is None:
            anchors = [(10, 13), (16, 30), (33, 23),
                       (30, 61), (62, 45), (59, 119),
                       (116, 90), (156, 198), (373, 326)]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

        self.features_extractor = Darknet53()
        self.anchors_network_s1 = AnchorsConv(1024, 1024)
        self.anchors_network_s2 = AnchorsConv(768, 512)
        self.anchors_network_s3 = AnchorsConv(384, 256)

        self.pre_anchors_network_2 = nn.Sequential(
            OrderedDict([
                (f"pre_anchors_network2_conv", nn.Conv2d(512, 256, 1, bias=False)),
                (f"pre_anchors_network2_batch_norm", nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)),
                (f"pre_anchors_network2_leaky", nn.LeakyReLU(0.1)),
                (f"pre_anchors_network2_upsample", Upsample(2))
            ])
        )

        self.pre_anchors_network_3 = nn.Sequential(OrderedDict([
            (f"pre_anchors_network3_conv", nn.Conv2d(256, 128, 1, bias=False)),
            (f"pre_anchors_network3_batch_norm", nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)),
            (f"pre_anchors_network3_leaky", nn.LeakyReLU(0.1)),
            (f"pre_anchors_network3_upsample", Upsample(2))
        ]))

        anchors_per_scale = len(anchors) // 3

        big_anchors = anchors[6:]
        medium_anchors = anchors[3:6]
        small_anchors = anchors[:3]
        self.yolo_layers = nn.ModuleList([
            YOLOLayer(big_anchors, n_classes, img_size),
            YOLOLayer(medium_anchors, n_classes, img_size),
            YOLOLayer(small_anchors, n_classes, img_size)
        ])

        self.yolo_conv_layers = nn.ModuleList([
            nn.Conv2d(1024, anchors_per_scale * (n_classes + 5), 1),
            nn.Conv2d(512, anchors_per_scale * (n_classes + 5), 1),
            nn.Conv2d(256, anchors_per_scale * (n_classes + 5), 1)
        ])

    def forward(self, x, targets=None):
        # x is a BxCxWxH tensor and targets is a T'x6 tensor, where T' is the number of targets in the whole batch
        # The 6 dimensions of the second tensor dimensions of targets is batch_index, class, x, y, w, h
        img_dim = x.shape[3]
        loss = 0

        ftrs = self.features_extractor(x)

        #  Big yolo conv
        anchors_net_layers_output1 = self.anchors_network_s1(ftrs[-1])
        anchors_net_output1 = anchors_net_layers_output1[-1]
        yolo_outputs_1 = self.yolo_conv_layers[0](anchors_net_output1)
        yolo_outputs_1, layer_loss = self.yolo_layers[0](yolo_outputs_1, targets, img_dim)
        loss += layer_loss

        # Medium yolo conv
        pre_anchors2_ftrs = self.pre_anchors_network_2(anchors_net_layers_output1[-2])

        pre_anchors2_ftrs = torch.cat([pre_anchors2_ftrs, ftrs[-2]], 1)

        anchors_net_layers_output2 = self.anchors_network_s2(pre_anchors2_ftrs)
        anchors_net_output2 = anchors_net_layers_output2[-1]
        yolo_outputs_2 = self.yolo_conv_layers[1](anchors_net_output2)
        yolo_outputs_2, layer_loss = self.yolo_layers[1](yolo_outputs_2, targets, img_dim)
        loss += layer_loss

        # Small yolo conv
        pre_anchors3_ftrs = self.pre_anchors_network_3(anchors_net_layers_output2[-2])

        pre_anchors3_ftrs = torch.cat([pre_anchors3_ftrs, ftrs[-3]], 1)

        anchors_net_layers_output3 = self.anchors_network_s3(pre_anchors3_ftrs)
        anchors_net_output3 = anchors_net_layers_output3[-1]
        yolo_outputs_3 = self.yolo_conv_layers[2](anchors_net_output3)
        yolo_outputs_3, layer_loss = self.yolo_layers[2](yolo_outputs_3, targets, img_dim)
        loss += layer_loss

        yolo_outputs = to_cpu(torch.cat([yolo_outputs_1, yolo_outputs_2, yolo_outputs_3], 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_yolov3_weights(self, weights_path):
        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

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
        ptr = self.load_conv_layer(self.anchors_network_s1.c1[0], current_ptr, weights, self.anchors_network_s1.c1[1])
        ptr = self.load_conv_layer(self.anchors_network_s1.c2[0], ptr, weights, self.anchors_network_s1.c2[1])
        ptr = self.load_conv_layer(self.anchors_network_s1.c3[0], ptr, weights, self.anchors_network_s1.c3[1])
        ptr = self.load_conv_layer(self.anchors_network_s1.c4[0], ptr, weights, self.anchors_network_s1.c4[1])
        ptr = self.load_conv_layer(self.anchors_network_s1.c5[0], ptr, weights, self.anchors_network_s1.c5[1])
        ptr = self.load_conv_layer(self.anchors_network_s1.c6[0], ptr, weights, self.anchors_network_s1.c6[1])

        ptr += self.yolo_conv_layers[0].bias.numel()
        ptr = self.load_conv_layer(self.pre_anchors_network_2[0], ptr, weights, self.pre_anchors_network_2[1])

        ptr = self.load_conv_layer(self.anchors_network_s2.c1[0], ptr, weights, self.anchors_network_s2.c1[1])
        ptr = self.load_conv_layer(self.anchors_network_s2.c2[0], ptr, weights, self.anchors_network_s2.c2[1])
        ptr = self.load_conv_layer(self.anchors_network_s2.c3[0], ptr, weights, self.anchors_network_s2.c3[1])
        ptr = self.load_conv_layer(self.anchors_network_s2.c4[0], ptr, weights, self.anchors_network_s2.c4[1])
        ptr = self.load_conv_layer(self.anchors_network_s2.c5[0], ptr, weights, self.anchors_network_s2.c5[1])
        ptr = self.load_conv_layer(self.anchors_network_s2.c6[0], ptr, weights, self.anchors_network_s2.c6[1])

        ptr += self.yolo_conv_layers[1].bias.numel()
        ptr = self.load_conv_layer(self.pre_anchors_network_3[0], ptr, weights, self.pre_anchors_network_3[1])

        ptr = self.load_conv_layer(self.anchors_network_s3.c1[0], ptr, weights, self.anchors_network_s3.c1[1])
        ptr = self.load_conv_layer(self.anchors_network_s3.c2[0], ptr, weights, self.anchors_network_s3.c2[1])
        ptr = self.load_conv_layer(self.anchors_network_s3.c3[0], ptr, weights, self.anchors_network_s3.c3[1])
        ptr = self.load_conv_layer(self.anchors_network_s3.c4[0], ptr, weights, self.anchors_network_s3.c4[1])
        ptr = self.load_conv_layer(self.anchors_network_s3.c5[0], ptr, weights, self.anchors_network_s3.c5[1])
        self.load_conv_layer(self.anchors_network_s3.c6[0], ptr, weights, self.anchors_network_s3.c6[1])

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
