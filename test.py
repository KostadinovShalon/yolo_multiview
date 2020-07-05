import json
from itertools import permutations

import numpy as np
import torch
import torch.utils.data

import evaluation
from yolov3.datasets import COCODataset, MVCOCODataset, COCODatasetFromMV
from yolov3.epipolar_geometry import compute_fundamental_matrix
from yolov3.test import evaluate_singleview, evaluate_multiview
from yolov3.utils.parser import get_parser_from_arguments
from yolov3.yolo import YOLOv3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_single_view(parser, valid_views=None):
    print("Runnning Test")
    dataset = COCODataset(parser.test["dir"],
                          annotations_file=parser.test["annotation_file"],
                          augment=False,
                          multiscale=False,
                          normalized_labels=parser.test["normalized"],
                          views=valid_views,
                          img_size=parser.img_size,
                          padding_value=1)
    # Initiate model
    model = YOLOv3(len(dataset.classes), anchors=parser.anchors).to(device)

    _, precision, recall, AP, f1, ap_class, detections = evaluate_singleview(
        dataset,
        model,
        parser.test["iou_thres"],
        parser.test["conf_thres"],
        parser.test["nms_thres"],
        parser.img_size,
        parser.workers,
        parser.test["weights_file"],
        parser.test["batch_size"],
        return_detections=True
    )

    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({dataset.class_indices[c]}) - AP: {AP[i]} - precision: {precision[i]}"
              f" - recall {recall[i]}")

    print(f"mAP: {AP.mean()}")

    return detections

    # json_file_name = os.path.join(parser.db_name, parser.test["json_file_output"])
    #
    # with open(json_file_name, 'w') as f:
    #     json.dump(detections, f)


def test_multi_view(parser, views=("A", "B"), weak_th=0.01, f_matrices=None):
    print("Getting Fundamental Matrices")
    with open(parser.train["annotation_file"], 'r') as f:
        coco = json.load(f)
    perms = permutations(views, 2)
    if f_matrices is None:
        f_matrices = {perm: compute_fundamental_matrix(coco, *perm, img_size=parser.img_size) for perm in perms}

    print("Runnning Test")
    dataset = MVCOCODataset(parser.test["dir"],
                            views=views,
                            annotations_file=parser.test["annotation_file"],
                            multiscale=False,
                            normalized_labels=parser.test["normalized"],
                            img_size=parser.img_size,
                            padding_value=1)

    # Initiate model
    model = YOLOv3(len(dataset.classes), anchors=parser.anchors).to(device)

    if parser.test["weights_file"] is not None:
        if parser.test["weights_file"].endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(parser.test["weights_file"])
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(parser.test["weights_file"]))
    model.eval()
    _, precision, recall, AP, f1, ap_class, detections = evaluate_multiview(dataset,
                                                                            model,
                                                                            parser.test["iou_thres"],
                                                                            parser.test["conf_thres"],
                                                                            weak_th,
                                                                            parser.test["nms_thres"],
                                                                            parser.img_size,
                                                                            parser.workers,
                                                                            f_matrices,
                                                                            views,
                                                                            bs=parser.test["batch_size"],
                                                                            return_detections=True)

    # for i, c in enumerate(ap_class):
    #     print(f"+ Class '{c}' ({dataset.get_cat_by_positional_id(c)}) - AP: {AP[i]} - precision: {precision[i]}"
    #           f" - recall {recall[i]}")
    #
    # print(f"mAP: {AP.mean()}")

    return detections

    # json_file_name = os.path.join(parser.db_name, parser.test["json_file_output"])
    #
    # with open(json_file_name, 'w') as f:
    #     json.dump(detections, f)


def test_single_view2(parser, views=("A", "B")):
    # with open(parser.train["annotation_file"], 'r') as f:
    #     coco = json.load(f)
    # perms = permutations(views, 2)
    # f_matrices = {perm: compute_fundamental_matrix(coco, *perm) for perm in perms}

    print("Runnning Test")
    dataset = COCODatasetFromMV(parser.test["dir"],
                                annotations_file=parser.test["annotation_file"],
                                multiscale=False,
                                augment=False,
                                normalized_labels=parser.test["normalized"],
                                img_size=parser.img_size,
                                views=views,
                                padding_value=1)

    # Initiate model
    model = YOLOv3(len(dataset.classes), anchors=parser.anchors).to(device)

    if parser.test["weights_file"] is not None:
        if parser.test["weights_file"].endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(parser.test["weights_file"])
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(parser.test["weights_file"]))
    model.eval()

    _, precision, recall, AP, f1, ap_class, detections = evaluate_singleview(
        dataset,
        model,
        parser.test["iou_thres"],
        parser.test["conf_thres"],
        parser.test["nms_thres"],
        parser.img_size,
        parser.workers,
        parser.test["weights_file"],
        parser.test["batch_size"],
        return_detections=True
    )

    # for i, c in enumerate(ap_class):
    #     print(f"+ Class '{c}' ({dataset.get_cat_by_positional_id(c)}) - AP: {AP[i]} - precision: {precision[i]}"
    #           f" - recall {recall[i]}")
    #
    # print(f"mAP: {AP.mean()}")

    return detections

    # json_file_name = os.path.join(parser.db_name, parser.test["json_file_output"])
    #
    # with open(json_file_name, 'w') as f:
    #     json.dump(detections, f)


def get_matrices():
    return {('A', 'B'): (np.array([[-6.49846751e-08, -1.56208880e-06, 1.14940316e-02],
                                   [1.17595112e-06, 2.44847446e-09, -3.78010660e-04],
                                   [-1.12711827e-02, 4.77200618e-04, -5.44683005e-02]]), 1.6603079476861844,
                         1.7495329338440575),
            ('A', 'C'): (np.array([[7.86691766e-08, -4.81552317e-08, 1.12326149e-02],
                                   [-1.74047680e-07, 5.53150519e-10, -8.43242885e-05],
                                   [-1.11968573e-02, 1.22007945e-05, 3.97474437e-02]]), 2.2048092923259244,
                         2.565097515178005),
            ('A', 'D'): (np.array([[4.08555744e-07, 1.81332644e-06, 1.11409565e-02],
                                   [-2.55927583e-06, -4.52904561e-08, 8.50494413e-04],
                                   [-1.11671797e-02, -7.88544025e-04, 2.15120640e-02]]), 2.7979021349577984,
                         3.001831549683605),
            ('B', 'A'): (np.array([[2.77851201e-07, 2.45152503e-07, -1.11610603e-02],
                                   [3.06993664e-07, 1.30288182e-09, -1.58610753e-04],
                                   [1.08086662e-02, -7.99995648e-05, 9.98062086e-02]]), 1.6289989653149775,
                         1.7301251855491784),
            ('B', 'C'): (np.array([[1.11932969e-06, -1.37414149e-07, 1.15016153e-02],
                                   [1.42927145e-07, 2.44572599e-09, -2.52685953e-04],
                                   [-1.22008033e-02, 3.95969985e-05, 1.85226978e-01]]), 2.239984051665111,
                         2.6042149337998524),
            ('B', 'D'): (np.array([[2.56293076e-07, 9.78994693e-07, -1.17938428e-02],
                                   [-1.95587836e-06, -7.31813297e-09, 8.39294263e-04],
                                   [1.19268330e-02, -3.73480827e-04, -1.23162589e-01]]), 2.6501604873094498,
                         2.858273176154957),
            ('C', 'A'): (np.array([[3.77039309e-07, -6.13433823e-07, 1.16440555e-02],
                                   [4.76962050e-07, -1.86396244e-09, -2.26528697e-04],
                                   [-1.17284017e-02, 3.36806103e-04, -5.13016027e-02]]), 2.251385866550074,
                         2.549124342333987),
            ('C', 'B'): (np.array([[1.32133850e-06, -1.26440512e-06, 1.27150924e-02],
                                   [7.32953752e-07, -1.40831164e-08, -1.93422850e-04],
                                   [-1.33937526e-02, 5.93185105e-04, -6.38912606e-03]]), 2.42059666884262,
                         2.8176204157066462),
            ('C', 'D'): (np.array([[1.18750364e-06, -5.10632353e-06, -1.13654553e-02],
                                   [-7.72238828e-07, 4.75091871e-08, 3.24867873e-04],
                                   [1.25783491e-02, 1.45675338e-03, -4.76066850e-01]]), 3.2345907901331867,
                         3.77643324764503),
            ('D', 'A'): (np.array([[-4.08547358e-07, 3.10930750e-06, -1.22545350e-02],
                                   [-1.26199131e-06, -4.16882860e-09, 4.51640724e-04],
                                   [1.19646100e-02, -1.07374162e-03, 1.56183785e-01]]), 2.6920535819199545,
                         3.001311290247895),
            ('D', 'B'): (np.array([[9.95249125e-07, -2.18253695e-06, 1.20051047e-02],
                                   [2.82684190e-06, 2.29647453e-09, -1.00687023e-03],
                                   [-1.29112013e-02, 7.68388312e-04, 1.89629807e-01]]), 2.6259162430419707,
                         2.8611247046812647),
            ('D', 'C'): (np.array([[7.72736407e-07, 2.14740881e-06, 1.12969209e-02],
                                   [2.45640654e-07, -3.35544658e-08, -2.61678631e-04],
                                   [-1.25026678e-02, -6.98602881e-04, 3.73087345e-01]]), 3.2444873100528278,
                         4.182939012480628)}


if __name__ == '__main__':
    gt_path = "/home/brian/Documents/datasets/new_smith_full/db4_test.json"
    p = get_parser_from_arguments()
    # detections = test_multi_view(parser, ("A", "B", "C", "D"), weak_th=0.01, f_matrices=get_matrices())
    preds = test_single_view2(p, ("A", "B", "C", "D"))
    stats = evaluation.evaluate(gt_path, preds, "conf_db4_sv.png")
    # json_file_name = os.path.join(parser.db_name, parser.test["json_file_output"])
    #
    # with open(json_file_name, 'w') as f:
    #     json.dump(detections, f)
