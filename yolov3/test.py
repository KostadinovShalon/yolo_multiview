import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from yolov3.utils.boxes import xywh2xyxy, non_max_suppression, rescale_boxes, mv_filtering
from yolov3.utils.statistics import get_batch_statistics, ap_per_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_singleview(dataset, model, iou_thres, conf_thres, nms_thres, img_size,
                        workers, weights_path=None, bs=1, return_detections=False):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=False,
        num_workers=workers,
        collate_fn=dataset.collate_fn
    )
    if weights_path is not None:
        if weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(weights_path))
    model.eval()

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    detections = []
    evaluation_loss = 0
    for batch_i, (img_paths, img_ids, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # Extract labels
        if targets.shape[0] > 0:
            labels += targets[:, 1].tolist()
        # Rescale target

        if device:
            imgs, targets = imgs.to(device), targets.to(device)
        imgs.requires_grad = False

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        # evaluation_loss += loss.item()
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        if return_detections:
            for img_path, img_id, dets in zip(img_paths, img_ids, outputs):
                if dets is not None:
                    for det in dets:
                        w, h = Image.open(img_path).convert('RGB').size
                        d = det.clone()
                        d[:4] = rescale_boxes(d[:4].unsqueeze(0), img_size, (h, w)).squeeze()
                        d = d.tolist()
                        detections.append({
                            "image_id": img_id,
                            "file_name": img_path.rsplit('/', 1)[1],
                            "category_id": dataset.class_indices[int(d[-1])],
                            "bbox": [d[0], d[1], d[2] - d[0], d[3] - d[1]],
                            "score": d[-2]
                        })

        sample_metrics += get_batch_statistics(outputs, targets.cpu(), iou_threshold=iou_thres)
    evaluation_loss /= len(dataloader)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    p, r, ap, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    if return_detections:
        return evaluation_loss, p, r, ap, f1, ap_class, detections
    else:
        return evaluation_loss, p, r, ap, f1, ap_class


def evaluate_multiview(dataset, model, iou_thres, conf_thres, weak_conf_thres, nms_thres, img_size, score_th,
                       workers, f_matrices, views, bs=1, return_detections=False):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=False,
        num_workers=workers,
        collate_fn=dataset.collate_fn
    )
    model.eval()

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    detections = []
    evaluation_loss = 0
    for batch_i, (img_paths, img_ids, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        view_outputs = {v: None for v in views}
        imgs, targets = imgs.to(device), targets.to(device)
        for i, v in enumerate(views):
            view_imgs = imgs[:, i, ...]
            view_imgs.requires_grad = False
            with torch.no_grad():
                view_outputs[v] = model(view_imgs)
                # view_outputs[v] = non_max_suppression(view_outputs[v], conf_thres=conf_thres, nms_thres=nms_thres)
            if targets.shape[0] > 0:
                view_targets = targets[:, i, :]
                labels += view_targets[:, 1].tolist()
                view_targets[:, 2:] = xywh2xyxy(view_targets[:, 2:])
                view_targets[:, 2:] *= img_size

        view_outputs = mv_filtering(view_outputs, f_matrices, conf_thres=conf_thres,
                                    nms_thres=nms_thres, score_th=score_th)
        for i, v in enumerate(views):
            if targets.shape[0] > 0:
                sample_metrics += get_batch_statistics(view_outputs[v], targets[:, i, :].cpu(), iou_threshold=iou_thres)
            if return_detections:
                for img_path, img_group_id, dets in zip(img_paths, img_ids, view_outputs[v]):
                    if dets is not None:
                        for det in dets:
                            w, h = Image.open(img_path[i]).convert('RGB').size
                            d = det.clone()
                            d[:4] = rescale_boxes(d[:4].unsqueeze(0), img_size, (h, w)).squeeze()
                            d = d.tolist()
                            detections.append({
                                "image_group_id": img_group_id,
                                "file_name": img_path[i].rsplit('/', 1)[1],
                                "category_id": dataset.class_indices[int(d[-1])],
                                "bbox": [d[0], d[1], d[2] - d[0], d[3] - d[1]],
                                "score": d[4] * d[5]
                            })

    # Concatenate sample statistics
    # true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    # # ordered_labels = []
    # # for lbl in labels.values():
    # #     ordered_labels.extend(lbl)
    p, r, ap, f1, ap_class = 0, 0, 0, 0, 0
    if return_detections:
        return evaluation_loss, p, r, ap, f1, ap_class, detections
    else:
        return evaluation_loss, p, r, ap, f1, ap_class


def evaluate_multiview_as_single_view(dataset, model, iou_thres, conf_thres, nms_thres, img_size,
                                      workers, views, bs=1, return_detections=False):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=False,
        num_workers=workers,
        collate_fn=dataset.collate_fn
    )
    model.eval()

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    detections = []
    evaluation_loss = 0
    for batch_i, (img_paths, img_ids, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        imgs, targets = imgs.to(device), targets.to(device)

        sv_targets = []
        sv_imgs = []

        for i in range(imgs.size(0)):
            im_group = imgs[i]
            for j, v in enumerate(views):
                im = im_group[j]
                im.requires_grad = False
                sv_imgs.append(im)
                im_targets = targets[targets[:, :, 0] == i].view(-1, len(views), 6)
                view_targets = im_targets[:, j, :]
                labels += view_targets[:, 1].tolist()
                view_targets[:, 2:] = xywh2xyxy(view_targets[:, 2:])
                view_targets[:, 2:] *= img_size
                view_targets[:, 0] = j + len(views) * i
                sv_targets.append(view_targets)

        # for i, v in enumerate(views):
        #     view_targets = targets[:, i, :]
        #     labels += view_targets[:, 1].tolist()
        #     view_imgs = imgs[:, i, ...]
        #     view_imgs.requires_grad = False
        #     sv_imgs.append(view_imgs)
        #     view_targets[:, 2:] = xywh2xyxy(view_targets[:, 2:])
        #     view_targets[:, 2:] *= img_size
        #     view_targets[:, 0] = i * len(view_imgs) + view_targets[:, 0]
        #     sv_targets.append(view_targets)
        sv_imgs = torch.stack(sv_imgs)
        sv_targets = torch.cat(sv_targets)
        with torch.no_grad():
            outputs = model(sv_imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        # view_outputs = mv_non_max_suppression(view_outputs, f_matrices, conf_thres=conf_thres,
        #                                       weak_conf_thres=weak_conf_thres,
        #                                       nms_thres=nms_thres,
        #                                       distance_margin=distance_margin)
        sample_metrics += get_batch_statistics(outputs, sv_targets.cpu(), iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    p, r, ap, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    if return_detections:
        return evaluation_loss, p, r, ap, f1, ap_class, detections
    else:
        return evaluation_loss, p, r, ap, f1, ap_class
