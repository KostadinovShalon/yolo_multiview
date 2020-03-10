import glob
import json
import os
from itertools import groupby
from typing import Union, List, Dict
import copy

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from yolov3.utils.geometry import *
from yolov3.utils.networks import horizontal_flip

import random

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def has_file_allowed_extension(filename: str, extensions: Union[Tuple, str]):
    """Checks if a file is an allowed extension.

    :param filename: path to a file
    :param extensions: extensions to consider (lowercase)

    :returns: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    :param filename: path to a file

    :returns: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def is_file(filename):
    """Checks if a file is an allowed image extension.

    :param filename: path to a file

    :returns: True if the filename ends with a known image extension
    """
    return os.path.isfile(filename)


# noinspection PyTypeChecker
def _get_annotations(json_file):
    """
    Gets the annotations from a VGG Json File
    :param json_file: vgg json file
    :return: a list with the annotation regions from the json file
    """
    with open(json_file, 'r') as f:
        raw_annotations = json.load(f)
    annotations = [dict(filename=ann['filename'], regions=ann['regions']) for _, ann in raw_annotations.items()]
    return annotations
#
#
# class COCODataset(Dataset):
#
#     def __init__(self, root: str,
#                  annotations_file: str,
#                  img_size=416,
#                  augment=True,
#                  multiscale=True,
#                  normalized_labels=True,
#                  partition=None,
#                  val_split=0.2,
#                  seed=None):
#         """
#         Dataset for files with the structure
#
#         - root
#            |  - image1.jpg
#            |  - image2.jpg
#            |  - ...
#
#         :param root: root directory path where images are
#         :param img_size: rescaled image size
#         :param augment: boolean indicating if data augmentation is going to be applied
#         :param multiscale: indicates if multi-scale must be used
#         :param normalized_labels: indicates if labels in the annotation file are normalized
#         :param partition: choose between train and val, None if there training and validation directories are separated.
#             If None, then it means validation and training are in different folders and all data is taken
#         :param val_split: split of the data for validation
#         :param seed: random seed
#         """
#         self.root = root
#         self.anns_file = annotations_file
#
#         with open(self.anns_file, 'r') as f:
#             coco_file = json.load(f)
#
#         self.classes = {int(category['id']): category['name'] for category in coco_file['categories']}
#         self.c = [i for i, _ in self.classes.items()]
#         self.c.sort()
#         self._actual_indices = {k: i for i, k in enumerate(self.c)}
#         self.imgs = [{"id": int(img['id']),
#                       "file_name": img['file_name']} for img in coco_file['images']]
#         self.anns = [{"image_id": ann['image_id'],
#                       "category_id": ann['category_id'],
#                       "bbox": ann['bbox']} for ann in coco_file['annotations']]
#
#         if partition is not None:
#             if seed is not None:
#                 random.seed(seed)
#             random.shuffle(self.imgs)
#
#             total_imgs = len(self.imgs)
#             total_partition_val = int(total_imgs * val_split)
#
#             self.imgs = self.imgs[:total_partition_val] if partition == "val" else self.imgs[total_partition_val:]
#
#         self.img_size = img_size
#         self.augment = augment
#         self.multiscale = multiscale
#         self.normalized_labels = normalized_labels
#         self.min_size = self.img_size - 3 * 32
#         self.max_size = self.img_size + 3 * 32
#         self.batch_count = 0
#
#     def get_cat_by_positional_id(self, positional_id):
#         cat_id = self.c[positional_id]
#         return self.classes[cat_id]
#
#     def anns_to_bounding_boxes(self, anns, img):
#         """
#         Converts a region dict to bounding boxes
#         :param anns: lists of anns dictionaries
#         :param img: original image
#         :return: bounding boxes
#         """
#         _, h, w = img.shape
#
#         # Pad to square resolution
#         img, pad = pad_to_square(img, 0)
#         _, padded_h, padded_w = img.shape
#         h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
#
#         boxes = None
#         if len(anns) > 0:
#             number_of_annotations = len(anns)
#             boxes = torch.zeros((number_of_annotations, 6))
#
#             for c, ann in enumerate(anns):
#                 bbox = ann["bbox"]
#
#                 # This adjustment is done when having bounding boxes outside the image boundaries
#                 xb, yb, wb, hb = bbox[0], bbox[1], bbox[2], bbox[3]
#
#                 if xb < 0:
#                     wb += xb
#                     xb = 0
#                 if yb < 0:
#                     hb += yb
#                     yb = 0
#                 if xb + wb > w:
#                     wb = w - xb
#                 if yb + hb > h:
#                     hb = h - yb
#
#                 # Unpadded and unscaled image
#                 # IMPORTANT! THIS IS ONLY FOR COCO DATASET
#                 x1 = xb * w_factor
#                 x2 = (xb + wb) * w_factor
#                 y1 = yb * h_factor
#                 y2 = (yb + hb) * h_factor
#                 # Adding paddding
#                 x1 += pad[0]
#                 y1 += pad[2]
#                 x2 += pad[1]
#                 y2 += pad[2]
#
#                 # Obtaining x, y, w, h
#                 boxes[c, 1] = ann["category_id"]
#                 boxes[c, 2] = (x1 + x2) / 2 / padded_w
#                 boxes[c, 3] = (y1 + y2) / 2 / padded_h
#                 boxes[c, 4] = (x2 - x1) * w_factor / padded_w
#                 boxes[c, 5] = (y2 - y1) * h_factor / padded_h
#         boxes[:, 1] = torch.tensor(list(map(self._actual_indices.get, boxes[:, 1].tolist())),
#                                    device=boxes.device)
#         return img, boxes
#
#     def __getitem__(self, index: int) -> Tuple[str, int, torch.Tensor, torch.Tensor]:
#
#         img = self.imgs[index]
#         anns = [a for a in self.anns if a["image_id"] == img["id"]]
#         img_path = os.path.join(self.root, img["file_name"])
#         img_id = img["id"]
#
#         img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
#         if len(img.shape) != 3:
#             img = img.unsqueeze(0)
#             img = img.expand((3, img.shape[1:]))
#
#         img, targets = self.anns_to_bounding_boxes(anns, img)
#         if self.augment:
#             if np.random.random() < 0.5:
#                 img, targets = horizontal_flip(img, targets)
#         return img_path, img_id, img, targets
#
#     def collate_fn(self, batch) -> Tuple[Tuple[str], Tuple[int], torch.Tensor, torch.Tensor]:
#         paths, img_ids, imgs, targets = list(zip(*batch))
#         targets = [boxes for boxes in targets if boxes is not None]
#
#         for i, boxes in enumerate(targets):
#             boxes[:, 0] = i
#         targets = torch.cat(targets, 0)
#         if self.multiscale and self.batch_count % 10 == 0:
#             self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
#
#         imgs = torch.stack([resize(img, self.img_size) for img in imgs])
#         self.batch_count += 1
#         return paths, img_ids, imgs, targets
#
#     def __len__(self):
#         return len(self.imgs)


class ImageFolder(Dataset):
    """
    Simple Dataset obtaining all files in a folder.
    """

    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.files = [f for f in self.files if is_file(f) and is_image_file(f)]
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        elif img.shape[0] == 1:
            img = img.squeeze()
            img = img.expand((3, img.shape[1:]))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


def get_common_string(name: str, view_suffices: Union[Tuple, List]):
    base_name = name.rsplit('.', 1)[0]
    for suffix in view_suffices:
        if base_name.endswith(suffix):
            return base_name.rsplit(suffix, 1)[0]
    return None


class ResizeWithPadding:

    def __init__(self, output_size, mode='constant'):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, tuple):
            assert len(output_size) == 2
        assert mode in ['constant', 'random']
        self.mode = mode
        self.output_size = output_size

    def __call__(self, sample):
        H, W = self.output_size if isinstance(self.output_size, tuple) \
                   else self.output_size, self.output_size
        w, h = sample.size
        r_0 = h / w
        r_objective = H / W
        if r_objective > r_0:
            p = w * r_objective - h
            sample = F.pad(sample, padding=(0, int(p // 2)))
        elif r_objective < r_0:
            p = (h / r_objective) - w
            sample = F.pad(sample, padding=(int(p // 2), 0))
        sample = F.resize(sample, (H, W))
        return sample


class ListDataset(Dataset):

    def __init__(self, root: str, view_suffix=('A', 'B'), img_size=416, transform=None, split='train',
                 negative_proposals=8, ignore_class='negative'):
        """
        Dataset for files with the structure

        - class1
        |  - train
           |  - image1.jpg
           |  - image2.jpg
           |  - ...
        |  - val
           |  - image1.jpg
           |  - image2.jpg
           |  - ...
        - class2
           |  - train...

        :param root: root directory path where data is present
        :param img_size: rescaled image size
        :param transform: pass
        :param split: choose between train and val
        """
        self.root = root
        self.transform = transform
        self.classes = self._find_classes()
        self.img_size = img_size
        self.imgs = dict()
        self.negative_images = []
        self.ignore_class = ignore_class
        self.negative_proposals = negative_proposals
        for i, cls in enumerate(self.classes):
            class_path = os.path.join(root, cls, split)
            imgs = [os.path.join(class_path, f) for f in os.listdir(class_path) if
                    is_file(os.path.join(class_path, f))
                    and is_image_file(f)]
            if cls == ignore_class:
                self.negative_images += imgs
            else:
                imgs.sort()
                imgs = [tuple(g) for k, g in groupby(imgs,
                                                     lambda a: get_common_string(a, view_suffix)) if k is not None]
                imgs = [img_tuple for img_tuple in imgs if len(img_tuple) == len(view_suffix)]
                self.imgs[cls] = imgs

    def __getitem__(self, index: int):
        cls = None
        original_index = index
        for k, v in self.imgs.items():
            if index < len(v):
                cls = k
                break
            else:
                index = index - len(v)
        positive_images = self.imgs[cls][index]
        n_negatives = len(self) + len(self.negative_images)
        negatives = random.sample(range(n_negatives), self.negative_proposals)
        negative_images = []
        for negative in negatives:
            negative_class = None
            i = negative + 1 if negative == original_index else negative
            for k, v in self.imgs.items():
                if i < len(v):
                    negative_class = k
                    break
                else:
                    i = i - len(v)
            if negative_class is None:
                negative_images.append(self.negative_images[i])
            else:
                negative_images.append(self.imgs[negative_class][i][1])

        positive_images = [Image.open(img_path).convert('RGB') for img_path in positive_images]
        negative_images = [Image.open(img_path).convert('RGB') for img_path in negative_images]
        if self.transform:
            positive_images = [self.transform(positive_image) for positive_image in positive_images]
            negative_images = [self.transform(negative_image) for negative_image in negative_images]
        negative_images = torch.stack(negative_images)

        return positive_images[0], positive_images[1], negative_images

    def __len__(self):
        return sum(len(itms) for k, itms in self.imgs.items())


class MVCOCODataset(Dataset):
    def __init__(self, root: str,
                 annotations_file: str,
                 views=("A", "B"),
                 img_size=416,
                 multiscale=True,
                 normalized_labels=True,
                 partition=None,
                 val_split=0.2,
                 seed=None):
        """
        Dataset for files with the structure

        - root
           |  - image1.jpg
           |  - image2.jpg
           |  - ...

        :param root: root directory path where images are
        :param img_size: rescaled image size
        :param multiscale: indicates if multi-scale must be used
        :param normalized_labels: indicates if labels in the annotation file are normalized
        :param partition: choose between train and val, None if there training and validation directories are separated.
            If None, then it means validation and training are in different folders and all data is taken
        :param val_split: split of the data for validation
        :param seed: random seed
        """
        self.root = root
        self.anns_file = annotations_file

        with open(self.anns_file, 'r') as f:
            coco_file = json.load(f)

        self.classes = {int(category['id']): category['name'] for category in coco_file['categories']}
        self.c = [i for i, _ in self.classes.items()]
        self.c.sort()
        self._actual_indices = {k: i for i, k in enumerate(self.c)}
        self.imgs = [{"id": int(img['id']),
                      "file_name": img['file_name']} for img in coco_file['images']]

        self.mv_anns = [ann for ann in coco_file["annotations"] if all(k in ann["views"].keys() for k in views)]

        if partition is not None:
            if seed is not None:
                random.seed(seed)
            random.shuffle(self.mv_anns)

            total_anns = len(self.mv_anns)
            total_partition_val = int(total_anns * val_split)

            self.mv_anns = self.total_anns[:total_partition_val] if partition == "val" \
                else self.total_anns[total_partition_val:]

        self.img_size = img_size
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.views = views

    def get_cat_by_positional_id(self, positional_id):
        cat_id = self.c[positional_id]
        return self.classes[cat_id]

    def anns_to_bounding_boxes(self, anns, imgs):
        """
        Converts a region dict to bounding boxes
        :param anns: lists of anns dictionaries
        :param imgs: original images
        :return: bounding boxes with format dict, and each view label includes a tensor as a value, with dims:
                number of annotations x 7 -> (batch_id, ann_id, label, x, y, w, h)
        """
        cat_id = anns["category_id"]
        ann_id = anns["id"]
        boxes = {}
        for k, view_ann in anns["views"].items():
            img = imgs[k]
            _, h, w = img.shape

            # Pad to square resolution
            padded_img, pad = pad_to_square(img, 0)
            imgs[k] = padded_img
            _, padded_h, padded_w = padded_img.shape
            h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

            view_boxes = None
            if len(view_ann) > 0:
                number_of_annotations = len(view_ann)
                view_boxes = torch.zeros((number_of_annotations, 7))

                for c, ann in enumerate(view_ann):
                    bbox = ann["bbox"]

                    # This adjustment is done when having bounding boxes outside the image boundaries
                    xb, yb, wb, hb = bbox[0], bbox[1], bbox[2], bbox[3]

                    if xb < 0:
                        wb += xb
                        xb = 0
                    if yb < 0:
                        hb += yb
                        yb = 0
                    if xb + wb > w:
                        wb = w - xb
                    if yb + hb > h:
                        hb = h - yb

                    # Unpadded and unscaled image
                    # IMPORTANT! THIS IS ONLY FOR COCO DATASET
                    x1 = xb * w_factor
                    x2 = (xb + wb) * w_factor
                    y1 = yb * h_factor
                    y2 = (yb + hb) * h_factor
                    # Adding paddding
                    x1 += pad[0]
                    y1 += pad[2]
                    x2 += pad[1]
                    y2 += pad[2]

                    # Obtaining x, y, w, h
                    view_boxes[c, 1] = cat_id
                    view_boxes[c, 2] = (x1 + x2) / 2 / padded_w
                    view_boxes[c, 3] = (y1 + y2) / 2 / padded_h
                    view_boxes[c, 4] = (x2 - x1) * w_factor / padded_w
                    view_boxes[c, 5] = (y2 - y1) * h_factor / padded_h
                    view_boxes[c, 6] = ann_id
            view_boxes[:, 1] = torch.tensor(list(map(self._actual_indices.get, view_boxes[:, 1].tolist())),
                                            device=view_boxes.device)
            boxes[k] = view_boxes
        return imgs, boxes

    def __getitem__(self, index: int) -> Tuple[Dict, Dict]:

        anns = self.mv_anns[index]
        imgs = {}
        valid_anns = copy.deepcopy(anns)
        for view, ann_view in anns["views"].items():
            if view in self.views:
                imgs[view] = next(im for im in self.imgs if im["id"] == ann_view[0]["image_id"])
            else:
                del valid_anns["views"][view]
        img_paths = {view: os.path.join(self.root, img["file_name"]) for view, img in imgs.items()}
        img_ids = {view: img["id"] for view, img in imgs.items()}

        imgs = {}
        for view, img_path in img_paths.items():
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
            if len(img.shape) != 3:
                img = img.unsqueeze(0)
                img = img.expand((3, img.shape[1:]))
            imgs[view] = img

        imgs, targets = self.anns_to_bounding_boxes(valid_anns, imgs)
        # At this point, imgs is a dictionary of the form
        # {
        #     "A": torch.tensor,
        #     "B": torch.tensor, ...
        # }
        # and targets the same where the tensor is anns x 6. It'll usually be anns = 1
        #
        return imgs, targets

    def collate_fn(self, batch) -> Dict:
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        imgs, targets = list(zip(*batch))  # These are lists of dictionaries
        output = {}
        for i, (img, target) in enumerate(zip(imgs, targets)):
            # img and target are dictionaries with the form of __getitem__ output
            for k, im in img.items():
                ann = target[k]
                if k not in output.keys():
                    output[k] = {"images": [], "targets": []}
                output[k]["images"].append(im)
                if ann is not None:
                    ann[:, 0] = i
                    output[k]["targets"].append(ann)
        for k in output.keys():
            output[k]["targets"] = torch.cat(output[k]["targets"], 0)
            output[k]["images"] = torch.stack([resize(img, self.img_size) for img in output[k]["images"]])

        self.batch_count += 1
        return output

    def __len__(self):
        return len(self.mv_anns)
