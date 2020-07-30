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

    parser.inference_dir = os.path.join(parser.inference_dir, "mv")

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

# This is a test
# def detect_oneset_multiview(model, views, F, F2, parser, image_a_path, image_b_path):
#     image_a = transforms.ToTensor()(Image.open(image_a_path).convert('RGB'))
#     image_a, _ = pad_to_square(image_a, 1)
#     image_a = resize(image_a, parser.img_size)
#
#     image_b = transforms.ToTensor()(Image.open(image_b_path).convert('RGB'))
#     image_b, _ = pad_to_square(image_b, 1)
#     image_b = resize(image_b, parser.img_size)
#
#     with open(parser.inference["annotation_file"], 'r') as f:
#         coco = json.load(f)
#     categories = sorted(coco['categories'], key=lambda key: key['id'])
#     classes = [c['name'] for c in categories]
#
#     if parser.inference["weights_file"].endswith(".weights"):
#         model.load_darknet_weights(parser.inference["weights_file"])
#     else:
#         model.load_state_dict(torch.load(parser.inference["weights_file"]))
#
#     model.eval()
#
#     view_outputs = {k: None for k in views}
#     with torch.no_grad():
#         v0, v1 = views
#         image_a = image_a.to(device)
#         image_b = image_b.to(device)
#         view_outputs[v0] = model(image_a.unsqueeze(0)).squeeze()
#         view_outputs[v1] = model(image_b.unsqueeze(0)).squeeze()
#
#         for k in views:
#             view_outputs[k][..., :4] = xywh2xyxy(view_outputs[k][..., :4])
#
#         conf_t_dets_a = conf_thresholding(view_outputs[v0], 0.5)
#         conf_t_dets_b = conf_thresholding(view_outputs[v1], 0.5)
#
#         draw_detections(image_a_path, "im1_a.png", conf_t_dets_a, classes, parser.img_size, with_name=False)
#         draw_detections(image_b_path, "im1_b.png", conf_t_dets_b, classes, parser.img_size, with_name=False)
#
#         src_centres_a = copy.deepcopy(conf_t_dets_a)
#         src_centres_a[..., :4] = xyxy2xywh(src_centres_a[..., :4])
#         src_centres_a = np.concatenate((src_centres_a[:, :2], np.ones((len(src_centres_a), 1))), axis=1)
#
#         weaks_b = [None] * len(src_centres_a)
#
#         for i, (src_centre, src_pred) in enumerate(zip(src_centres_a, conf_t_dets_a)):
#             src_class = src_pred[-1].cpu().long().item()
#             dst_dets = conf_thresholding(view_outputs[v1], 1e-2, src_class)
#             if dst_dets is None:
#                 continue
#             f, error, std = F[src_class]
#             epiline = f @ src_centre.transpose()
#             A, B, C = epiline[0], epiline[1], epiline[2]
#             x = (dst_dets[:, 0] + dst_dets[:, 2]) / 2
#             y = (dst_dets[:, 1] + dst_dets[:, 3]) / 2
#             d = (A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)
#             distance_probs = prob_by_distance(d, error, std)
#             dst_dets[:, 4] = dst_dets[:, 4] * distance_probs
#             score_with_distance = dst_dets[:, 4] * dst_dets[:, 5]
#             import warnings
#             warnings.filterwarnings("ignore", category=UserWarning)
#             valid_weak_preds = dst_dets[score_with_distance > 0.5 * 0.05]
#             # If none are remaining => process next image
#             draw_detections(image_a_path, f"im2_a_single_{i}.png", conf_t_dets_a[i].unsqueeze(0), classes,
#                             parser.img_size, with_name=False)
#             if not valid_weak_preds.size(0):
#                 draw_detections(image_b_path, f"im2_b_epifil_{i}.png", None, classes, parser.img_size, f=f,
#                 with_name=False,
#                                 epipoints=torch.tensor(src_centre).unsqueeze(0))
#                 continue
#             # Object confidence times class confidence
#             # Sort by it
#
#             draw_detections(image_b_path, f"im2_b_epifil_{i}.png", valid_weak_preds, classes, parser.img_size,f=f,
#             epipoints=torch.tensor(src_centre).unsqueeze(0), with_name=False)
#             score_with_distance = valid_weak_preds[:, 4] * valid_weak_preds[:, 5]
#             valid_weak_preds = valid_weak_preds[(-score_with_distance).argsort()]
#             valid_weak_preds = torch.stack(nms(valid_weak_preds, 0.4))
#             draw_detections(image_b_path, f"im3_b_epifil_{i}.png", valid_weak_preds[0].unsqueeze(0), classes,
#             parser.img_size, f=f,  epipoints=torch.tensor(src_centre).unsqueeze(0), with_name=False)
#             weaks_b[i] = valid_weak_preds[0]
#
#         src_centres_b = copy.deepcopy(conf_t_dets_b)
#         src_centres_b[..., :4] = xyxy2xywh(src_centres_b[..., :4])
#         src_centres_b = np.concatenate((src_centres_b[:, :2], np.ones((len(src_centres_b), 1))), axis=1)
#         weaks_a = [None] * len(src_centres_b)
#
#         for i, (src_centre, src_pred) in enumerate(zip(src_centres_b, conf_t_dets_b)):
#             src_class = src_pred[-1].cpu().long().item()
#             dst_dets = conf_thresholding(view_outputs[v0], 1e-2, src_class)
#             if dst_dets is None:
#                 continue
#             f, error, std = F2[src_class]
#             epiline = f @ src_centre.transpose()
#             A, B, C = epiline[0], epiline[1], epiline[2]
#             x = (dst_dets[:, 0] + dst_dets[:, 2]) / 2
#             y = (dst_dets[:, 1] + dst_dets[:, 3]) / 2
#             d = (A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)
#             distance_probs = prob_by_distance(d, error, std)
#             dst_dets[:, 4] = dst_dets[:, 4] * distance_probs
#             score_with_distance = dst_dets[:, 4] * dst_dets[:, 5]
#             import warnings
#             warnings.filterwarnings("ignore", category=UserWarning)
#             valid_weak_preds = dst_dets[score_with_distance > 0.5 * 0.05]
#             # If none are remaining => process next image
#             draw_detections(image_b_path, f"im2_b_single_{i}.png", conf_t_dets_b[i].unsqueeze(0), classes,
#                             parser.img_size, with_name=False)
#             if not valid_weak_preds.size(0):
#                 draw_detections(image_a_path, f"im2_a_epifil_{i}.png", None, classes, parser.img_size, f=f,
#                 with_name=False,
#                                 epipoints=torch.tensor(src_centre).unsqueeze(0))
#                 continue
#             # Object confidence times class confidence
#             # Sort by it
#
#
#             score_with_distance = valid_weak_preds[:, 4] * valid_weak_preds[:, 5]
#             valid_weak_preds = valid_weak_preds[(-score_with_distance).argsort()]
#             valid_weak_preds = torch.stack(nms(valid_weak_preds, 0.4))
#             draw_detections(image_a_path, f"im3_a_epifil_{i}.png", valid_weak_preds[0].unsqueeze(0), classes,
#             parser.img_size, with_name=False,
#                             f=f, epipoints=torch.tensor(src_centre).unsqueeze(0))
#             weaks_a[i] = valid_weak_preds[0]
#
#         valid_a = []
#         valid_b = []
#         for j in range(conf_t_dets_a.shape[0]):
#             # is_valid = True
#             if weaks_b[j] is not None:
#                 valid_a.append(conf_t_dets_a[j])
#                 valid_b.append(weaks_b[j])
#         for j in range(conf_t_dets_b.shape[0]):
#             # is_valid = True
#             if weaks_a[j] is not None:
#                 valid_b.append(conf_t_dets_b[j])
#                 valid_a.append(weaks_a[j])
#
#         if len(valid_a) > 0:
#             valid_a = torch.stack(valid_a)
#
#             draw_detections(image_a_path, "im4_a.png", valid_a, classes, parser.img_size, with_name=False)
#             draw_detections(image_a_path, "im5_a.png", torch.stack(nms(valid_a, 0.4)), classes, parser.img_size,
#             with_name=False)
#         if len(valid_b) > 0:
#             valid_b = torch.stack(valid_b)
#
#             draw_detections(image_b_path, "im4_b.png", valid_b, classes, parser.img_size, with_name=False)
#             draw_detections(image_b_path, "im5_b.png", torch.stack(nms(valid_b, 0.4)), classes, parser.img_size,
#             with_name=False)

        # preds_a = torch.stack(nms(conf_t_dets_a, 0.4)) if conf_t_dets_a is not None else None
        # preds_b = torch.stack(nms(conf_t_dets_b, 0.4)) if conf_t_dets_b is not None else None
        # draw_detections(image_a_path, "image_a_1step_0.8_nms.png", preds_a, classes, parser.img_size)
        # draw_detections(image_b_path, "image_b_1step_0.8_nms.png", preds_b, classes, parser.img_size)
        #
        # score_t_dets_a = score_thresholding(view_outputs[v0], 0.1, -1)
        # score_t_dets_b = score_thresholding(view_outputs[v1], 0.1, -1)
        #
        # draw_detections(image_a_path, "image_a_2step_s0.1.png", score_t_dets_a, classes, parser.img_size)
        # draw_detections(image_b_path, "image_b_2step_s0.1.png", score_t_dets_b, classes, parser.img_size)
        #
        # epi_filtered_b, centres_a = epi_filtering(preds_a, view_outputs[v1], 0.1, f_matrices[(v0, v1)], 60, True)
        # epi_filtered_a, centres_b = epi_filtering(preds_b, view_outputs[v0], 0.1, f_matrices[(v1, v0)], 60, True)
        #
        # drawable_epi_a, drawable_epi_b = None, None
        # if epi_filtered_a is not None:
        #     drawable_epi_a = [e for e in epi_filtered_a if e is not None]
        #     drawable_epi_a = torch.cat(drawable_epi_a, dim=0) if len(drawable_epi_a) > 0 else None
        # if epi_filtered_b is not None:
        #     drawable_epi_b = [e for e in epi_filtered_b if e is not None]
        #     drawable_epi_b = torch.cat(drawable_epi_b, dim=0) if len(drawable_epi_b) > 0 else None
        #
        # draw_detections(image_a_path, "image_a_3step_epi_noclass.png", drawable_epi_a, classes, parser.img_size,
        #                 f=f_matrices[(v1, v0)], epipoints=centres_b)
        # draw_detections(image_b_path, "image_b_3step_epi_noclass.png", drawable_epi_b, classes, parser.img_size,
        #                 f=f_matrices[(v0, v1)], epipoints=centres_a)
        #
        # epi_filtered_b, centres_a = epi_filtering(preds_a, view_outputs[v1], 0.1, f_matrices[(v0, v1)], 60)
        # epi_filtered_a, centres_b = epi_filtering(preds_b, view_outputs[v0], 0.1, f_matrices[(v1, v0)], 60)
        #
        # drawable_epi_a, drawable_epi_b = None, None
        # if epi_filtered_a is not None:
        #     drawable_epi_a = [e for e in epi_filtered_a if e is not None]
        #     drawable_epi_a = torch.cat(drawable_epi_a, dim=0) if len(drawable_epi_a) > 0 else None
        # if epi_filtered_b is not None:
        #     drawable_epi_b = [e for e in epi_filtered_b if e is not None]
        #     drawable_epi_b = torch.cat(drawable_epi_b, dim=0) if len(drawable_epi_b) > 0 else None
        #
        # draw_detections(image_a_path, "image_a_4step_epi.png", drawable_epi_a, classes, parser.img_size,
        #                 f=f_matrices[(v1, v0)], epipoints=centres_b)
        # draw_detections(image_b_path, "image_b_4step_epi.png", drawable_epi_b, classes, parser.img_size,
        #                 f=f_matrices[(v0, v1)], epipoints=centres_a)
        #
        # # Eliminating detections with no match
        # if preds_a is not None and drawable_epi_b is not None:
        #     preds_a = [pred_a.unsqueeze(0) for pred_a, projs_b in zip(preds_a, epi_filtered_b) if projs_b is not None]
        #     epi_filtered_b = [e for e in epi_filtered_b if e is not None]
        # else:
        #     preds_a = []
        #     epi_filtered_b = []
        # if preds_b is not None and drawable_epi_a is not None:
        #     preds_b = [pred_b.unsqueeze(0) for pred_b, projs_a in zip(preds_b, epi_filtered_a) if projs_a is not None]
        #     epi_filtered_a = [e for e in epi_filtered_a if e is not None]
        # else:
        #     preds_b = []
        #     epi_filtered_a = []
        #
        # total_dets_a = preds_a + epi_filtered_a
        # total_dets_b = preds_b + epi_filtered_b
        # total_dets_a = torch.cat(total_dets_a, dim=0) if len(total_dets_a) > 0 else None
        # total_dets_b = torch.cat(total_dets_b, dim=0) if len(total_dets_b) > 0 else None
        #
        # draw_detections(image_a_path, "image_a_5step_all_1.png", total_dets_a, classes, parser.img_size)
        # draw_detections(image_b_path, "image_b_5step_all_1.png", total_dets_b, classes, parser.img_size)
        #
        # nms_filtered_a = torch.stack(nms(total_dets_a, 0.4)) if total_dets_a is not None else None
        # nms_filtered_b = torch.stack(nms(total_dets_b, 0.4)) if total_dets_b is not None else None
        # draw_detections(image_a_path, "image_a_5step_all_nms_1.png", nms_filtered_a, classes, parser.img_size)
        # draw_detections(image_b_path, "image_b_5step_all_nms_1.png", nms_filtered_b, classes, parser.img_size)

#
# def epi_filtering(src_preds, dst_dets, score_th, fundamental_matrix, distance_margin, no_class=False):
#     # Filter out confidence scores below threshold
#     if src_preds is None:
#         return None, None
#     src_centres = xyxy2xywh(src_preds.clone().detach())
#     src_centres = np.concatenate((src_centres[:, :2], np.ones((len(src_preds), 1))), axis=1)
#     epilines = fundamental_matrix @ src_centres.transpose()
#
#     weaks = [None] * len(src_centres)
#     # if dst_dets is None:
#     #     return None, src_centres
#     for i, (src_centre, src_pred, epiline) in enumerate(zip(src_centres, src_preds, epilines.transpose())):
#         src_class = src_pred[-1].cpu().long().item()
#         if no_class:
#             src_class = -1
#         dst_preds = score_thresholding(dst_dets, score_th, src_class)
#         if dst_preds is None:
#             continue
#         A, B, C = epiline[0], epiline[1], epiline[2]
#         d = np.abs(A * dst_preds[:, 0] + B * dst_preds[:, 1] + C) / np.sqrt(A ** 2 + B ** 2)
#         valid_weak_preds = dst_preds[d < distance_margin]
#         # If none are remaining => process next image
#         if not valid_weak_preds.size(0):
#             continue
#         # # Object confidence times class confidence
#         # valid_weak_preds = torch.stack(nms(valid_weak_preds, nms_thres))
#         # score = valid_weak_preds[:, 4] * valid_weak_preds[:, 5]
#         # # Sort by it
#         # valid_weak_preds = valid_weak_preds[(-score).argsort()]
#         weaks[i] = valid_weak_preds
#     return weaks, src_centres
