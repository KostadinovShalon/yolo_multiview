import json
from itertools import permutations

import torch.utils.data

from test import get_matrices
from yolov3.datasets import MVCOCODataset
from yolov3.detect import detect_singleview, detect_multiview
from yolov3.epipolar_geometry import compute_fundamental_matrix
from yolov3.utils.parser import get_parser_from_arguments
from yolov3.yolo import YOLOv3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference_single_view(views=("A", "B")):
    parser = get_parser_from_arguments()

    print("Runnning Inference")
    # Initiate model
    classes = ["firearm", "firearmpart", "knife", "camera", "ceramic_knife", "laptop"]
    model = YOLOv3(len(classes), anchors=parser.anchors).to(device)

    detect_singleview(
        model,
        parser,
        views,
        classes)


def inference_multi_view(parser, views=("A", "B"), f_matrices=None):
    """
    Performs inference on a multi-view data set
    :param parser: parser object
    :param views:
    :param f_matrices:
    :return:
    """
    print("Getting Fundamental Matrices")
    with open(parser.train["annotation_file"], 'r') as f:
        coco = json.load(f)
    perms = permutations(views, 2)
    if f_matrices is None:
        f_matrices = {perm: compute_fundamental_matrix(coco, *perm) for perm in perms}

    print("Runnning Test")
    dataset = MVCOCODataset(parser.test["dir"],
                            views=views,
                            annotations_file=parser.test["annotation_file"],
                            multiscale=False,
                            normalized_labels=parser.test["normalized"],
                            img_size=parser.img_size)

    # Initiate model
    model = YOLOv3(len(dataset.classes), anchors=parser.anchors).to(device)
    detect_multiview(model, dataset, f_matrices, parser, weak_conf_th=0.01)


# def inference_oneimage_multi_view(views=("A", "B")):
#     parser = get_parser_from_arguments()
#     print("Getting Fundamental Matrices")
#     with open(parser.train["annotation_file"], 'r') as f:
#         coco = json.load(f)
#     perms = permutations(views, 2)
#     f_matrices = {perm: compute_fundamental_matrix(coco, *perm) for perm in perms}
#
#     image_a_path = ''
#     image_b_path = ''
#
#     # Initiate model
#     model = YOLOv3(4, anchors=parser.anchors).to(device)
#     detect_oneset_multiview(model, views, f_matrices, parser, image_a_path, image_b_path)


if __name__ == '__main__':
    p = get_parser_from_arguments()
    inference_multi_view(p, views=("A", "B", "C", "D"), f_matrices=get_matrices())
