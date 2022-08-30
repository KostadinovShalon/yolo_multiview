from torch.utils.data import DataLoader
from tqdm import tqdm

from yolov3.datasets import *
from yolov3.utils.boxes import non_max_suppression, xywh2xyxy, mv_filtering
from yolov3.utils.visualization import draw_detections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dirs(inference_dir, classes):
    """
    Makes the following dirs:
    inference_dir
        - none/
        - class_1
            - none
            - class_2
            - class_3
            ...
        - class_2
            - none
            - class_1
            - class_3
            ...
        ...
    :param inference_dir: path of the inference directory
    :param classes: classes list
    """
    os.makedirs(os.path.join(inference_dir, "none"), exist_ok=True)
    for c in classes:
        os.makedirs(os.path.join(inference_dir, c), exist_ok=True)
        for k in classes:
            if k != c:
                os.makedirs(os.path.join(inference_dir, c, k), exist_ok=True)
        os.makedirs(os.path.join(inference_dir, c, 'none'), exist_ok=True)
        os.makedirs(os.path.join(inference_dir, 'none', c), exist_ok=True)


def save_no_gt(img_path, inference_dir, filename, classes, img_size, img_detection):
    """
    Draws and saves the image with its predictions to one of the following folders:
        - inference/none/ if no detection was obtained
        - inference/none/class_k if the k-th class was found
    :param img_path: image path
    :param inference_dir: inference directory
    :param filename: file name of the output image
    :param classes: class names list
    :param img_size: image size
    :param img_detection: tensor with image detections
    """
    if img_detection is None or len(img_detection) == 0:
        out_path = os.path.join(inference_dir, 'none', f"{filename}.png")
        draw_detections(img_path, out_path, img_detection, classes, img_size)
    else:
        detected_classes = img_detection.t()[-1]
        for detected_class in detected_classes:
            cls = classes[int(detected_class)]
            out_path = os.path.join(inference_dir, 'none', cls, f"{filename}.png")
            draw_detections(img_path, out_path, img_detection, classes, img_size)


def save_detection(img_path, img_detection, ground_truth, inference_dir, filename, classes, img_size):
    gt_classes = ground_truth.t()[1]
    for gt_class in gt_classes:
        if gt_class >= len(classes):
            save_no_gt(img_path, inference_dir, filename, classes, img_size, img_detection)
        else:
            g_cls = classes[int(gt_class)]
            if img_detection is None or len(img_detection) == 0:
                out_path = os.path.join(inference_dir, g_cls, 'none', f"{filename}.png")
                draw_detections(img_path, out_path, img_detection, classes, img_size, gt=ground_truth)
            else:
                detected_classes = img_detection.t()[-1]
                for detected_class in detected_classes:
                    cls = classes[int(detected_class)]
                    if cls == g_cls:
                        out_path = os.path.join(inference_dir, cls, f"{filename}.png")
                    else:
                        out_path = os.path.join(inference_dir, g_cls, cls, f"{filename}.png")
                    draw_detections(img_path, out_path, img_detection, classes, img_size, gt=ground_truth)


def detect_singleview(model, parser, views=None, classes=None):
    """
    Inference for single view data
    :param model: model object
    :param parser: parser object
    :param views: views to include in coco dataset
    :param classes: class names list. If parser points to a annotation file, classes will be taken from there. Otherwise
    it will be taken from this list. If no annotation file is provided and neither this parameter, classes will be
    taken from the model
    """
    annotations = None
    if parser.inference["annotation_file"] is not None:
        with open(parser.inference["annotation_file"], 'r') as f:
            coco = json.load(f)
        annotations = coco['annotations']
        categories = sorted(coco['categories'], key=lambda key: key['id'])
        classes = [c['name'] for c in categories]
    else:
        if classes is None:
            classes = model.classes
    save_structured = annotations is not None and parser.inference["save_structured"]
    if save_structured and classes is not None:
        make_dirs(parser.inference_dir, classes)

    if parser.inference["weights_file"].endswith(".weights"):
        model.load_darknet_weights(parser.inference["weights_file"])
    else:
        model.load_state_dict(torch.load(parser.inference["weights_file"]))

    model.eval()
    with_gt = parser.inference["with_gt"] and annotations is not None
    dataloader_params = {"batch_size": parser.inference["batch_size"], "shuffle": False,
                         "num_workers": parser.workers}
    if with_gt:
        dataset = COCODatasetFromMV(parser.inference["dir"],
                                    annotations_file=parser.inference["annotation_file"],
                                    augment=False,
                                    multiscale=False,
                                    normalized_labels=parser.inference["normalized"],
                                    img_size=parser.img_size,
                                    views=views,
                                    padding_value=1)
        dataloader_params["collate_fn"] = dataset.collate_fn
    else:
        dataset = ImageFolder(parser.inference["dir"], img_size=parser.img_size)
    dataloader_params["dataset"] = dataset
    dataloader = torch.utils.data.DataLoader(**dataloader_params)

    n_imgs = parser.inference["max_images"]
    img_counter = 0

    parser.inference_dir = os.path.join(parser.inference_dir, "sv")
    os.makedirs(parser.inference_dir, exist_ok=True)

    for data in tqdm(dataloader, desc="Detecting objects and saving images"):
        # Configure input
        targets = None
        gt = None
        if not with_gt:
            img_paths, imgs = data
        else:
            img_paths, _, imgs, targets = data
            targets = targets.to(device)

        imgs = imgs.to(device)
        # Get detections
        with torch.no_grad():
            detections = model(imgs)
            detections = non_max_suppression(detections, parser.inference["conf_thres"],
                                             parser.inference["nms_thres"])

        for batch_id, (img_path, img_detection) in enumerate(zip(img_paths, detections)):
            if targets is not None:
                t = [t for t in targets if t[0] == batch_id]
                if len(t) > 0:
                    gt = torch.stack(t)
            filename = img_path.split("/")[-1].split(".")[0]
            if save_structured and gt is not None:
                if targets.size(0) == 0:
                    save_no_gt(img_path, parser.inference_dir, filename, classes, parser.img_size, None)
                else:
                    save_detection(img_path, None, gt, parser.inference_dir, filename, classes, parser.img_size)
            else:
                out_path = os.path.join(parser.inference_dir, f"{filename}.png")
                draw_detections(img_path, out_path, None, classes, parser.img_size, gt=gt)

        img_counter += len(img_paths)
        if 0 < n_imgs <= img_counter:
            break


def detect_multiview(model, dataset, f_matrices, parser):
    annotations = None
    classes = parser.inference["classes"]
    if parser.inference["annotation_file"] is not None:
        with open(parser.inference["annotation_file"], 'r') as f:
            coco = json.load(f)
        annotations = coco['annotations']
        categories = sorted(coco['categories'], key=lambda key: key['id'])
        classes = [c['name'] for c in categories]
    save_structured = annotations is not None and parser.inference["save_structured"]
    with_gt = parser.inference["with_gt"] and annotations is not None

    dataloader_params = {"batch_size": parser.inference["batch_size"], "shuffle": False,
                         "num_workers": parser.workers}
    if with_gt:
        dataloader_params["collate_fn"] = dataset.collate_fn
    else:
        dataset = ImageFolder(parser.inference["dir"], img_size=parser.img_size)

    dataloader_params["dataset"] = dataset
    dataloader = torch.utils.data.DataLoader(**dataloader_params)

    if save_structured:
        make_dirs(parser.inference_dir, classes)

    if parser.inference["weights_file"].endswith(".weights"):
        model.load_darknet_weights(parser.inference["weights_file"])
    else:
        model.load_state_dict(torch.load(parser.inference["weights_file"]))

    model.eval()

    n_imgs = parser.inference["max_images"]
    img_counter = 0
    views = dataset.views
    for img_paths, _, imgs, targets in tqdm(dataloader, desc="Detecting objects and saving images"):
        # NOTE: THIS ONLY WORKS WITH MVCOCO DATASET
        # Get detections
        view_outputs = {v: None for v in views}
        imgs, targets = imgs.to(device), targets.to(device)

        for i, v in enumerate(views):
            view_imgs = imgs[:, i, ...]
            view_imgs.requires_grad = False
            view_targets = targets[:, i, :]
            with torch.no_grad():
                view_outputs[v] = model(view_imgs)
                # view_outputs[v] = non_max_suppression(view_outputs[v], conf_thres=conf_thres, nms_thres=nms_thres)
            view_targets[:, 2:] = xywh2xyxy(view_targets[:, 2:])
            view_targets[:, 2:] *= parser.img_size

        detections = mv_filtering(view_outputs, f_matrices, conf_thres=parser.inference["conf_thres"],
                                  p_value=parser.inference["p_value"],
                                  nms_thres=parser.inference["nms_thres"])

        for j, v_img_paths in enumerate(img_paths):
            for i, v in enumerate(views):
                v_img_path = v_img_paths[i]
                filename = v_img_path.split("/")[-1].split(".")[0]
                out_path = os.path.join(parser.inference_dir, f"{filename}.png")
                draw_detections(v_img_path, out_path, detections[v][j], classes, parser.img_size)

        img_counter += len(img_paths)
        if 0 < n_imgs <= img_counter:
            break
