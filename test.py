import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from yolov3.utils.boxes import xywh2xyxy, non_max_suppression, remove_not_projected_boxes, xyxy2xywh
from yolov3.utils.networks import to_cpu
from yolov3.utils.statistics import get_batch_statistics, ap_per_class


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(dataset, model, iou_thres, conf_thres, nms_thres, img_size,
             workers, weights_path=None, bs=1):  # return_detections=False):

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
    for batch_i, (imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        for k, view_targets in targets.items():
            # Extract labels
            labels += view_targets[:, 1].tolist()
            # Rescale target
            view_targets[:, 2:] = xywh2xyxy(view_targets[:, 2:])
            view_targets[:, 2:] *= img_size

            if device:
                imgs[k] = imgs[k].to(device)
            imgs[k].requires_grad = False

        with torch.no_grad():
            outputs = model(imgs)
            for k, output in outputs.items():
                outputs[k] = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)
            # Removing projections
            # for m in outputs.keys():
            #     for n in outputs.keys():
            #         if m != n:
            #             for p, (output_n, output_m) in enumerate(zip(outputs[n], outputs[m])):
            #                 if output_n is not None and output_m is not None:
            #                     projections = model.get_projections(xyxy2xywh(output_n[:, :4]).to(device) / img_size, (n, m))
            #                     projections = torch.cat((projections * img_size, output_n[:, 4:].to(device)), 1)
            #                     prediction_box = xyxy2xywh(output_m[:, :4])
            #                     prediction = torch.cat([prediction_box, output_m[:, 4:]], 1).to(device)
            #                     outputs[m][p] = remove_not_projected_boxes(prediction, projections)
            #                     if outputs[m][p] is not None:
            #                         outputs[m][p] = to_cpu(outputs[m][p])
            #                 else:
            #                     outputs[m][p] = None
        # TODO: print json output file
        # for img_path, img_id, dets in zip(file_names, img_ids, outputs):
        #     if dets is not None:
        #         for det in dets:
        #             w, h = Image.open(img_path).convert('RGB').size
        #             d = det.clone()
        #             d[:4] = rescale_boxes(d[:4].unsqueeze(0), img_size, (h, w)).squeeze()
        #             d = d.tolist()
        #             detections.append({
        #                 "image_id": img_id,
        #                 "category_id": dataset.c[int(d[-1])],
        #                 "bbox": [d[0], d[1], d[2] - d[0], d[3] - d[1]],
        #                 "score": d[-2]
        #             })

        for k in targets.keys():
            sample_metrics += get_batch_statistics(outputs[k], targets[k], iou_threshold=iou_thres)

    # Concatenate sample statistics
    if len(sample_metrics) == 0:
        return 0, 0, 0, 0, [0]
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    #  return precision, recall, AP, f1, ap_class
    return ap_per_class(true_positives, pred_scores, pred_labels, labels)
