import copy
import glob
import json
import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from yolov3.utils.geometry import *
from yolov3.utils.networks import horizontal_flip

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    :param filename: path to a file
    :param extensions: (tuple of strings) extensions to consider (lowercase)

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


class _COCODataset(Dataset):

    def __init__(self, root: str,
                 annotations_file: str,
                 img_size=416,
                 augment=True,
                 multiscale=True,
                 normalized_labels=True,
                 partition=None,
                 val_split=0.2,
                 seed=None,
                 padding_value=1):
        """
        Dataset for coco-annotated data

        :param root: images root directory path
        :param img_size: rescaled image size (int)
        :param augment: boolean to apply data augmentation (horizontal flipping)
        :param multiscale: boolean to use multi-scale
        :param normalized_labels: indicates if labels in the annotation file are normalized
        :param padding_value: padding value to add
        """
        self.root = root
        self.anns_file = annotations_file

        with open(self.anns_file, 'r') as f:
            self._coco_file = json.load(f)

        self.classes = {int(category['id']): category['name'] for category in self._coco_file['categories']}
        self.class_indices = list(self.classes.keys())
        self.class_indices.sort()
        self.classes = [self.classes[c] for c in self.class_indices]

        self.imgs = self._get_images()
        self.anns = self._get_annotations()
        self._split()

        self.img_size = img_size
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.padding_value = padding_value
        self.partition = partition
        self.val_split = val_split
        self.seed = seed
        # self.wts = self._get_class_weights()

    def _get_images(self):
        return [{"id": int(img['id']),
                 "file_name": img['file_name']} for img in self._coco_file['images']]

    def _get_annotations(self):
        return [{"image_id": ann['image_id'],
                 "category_id": ann['category_id'],
                 "bbox": ann['bbox']} for ann in self._coco_file['annotations']]

    def _split(self):
        if self.partition is not None:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self.imgs)

            total_imgs = len(self.imgs)
            total_partition_val = int(total_imgs * self.val_split)

            self.imgs = self.imgs[:total_partition_val] if self.partition == "val" else self.imgs[total_partition_val:]

    def anns_to_bounding_boxes(self, anns, img):
        """
        Converts a region dict to bounding boxes
        :param anns: lists of anns dictionaries
        :param img: original image
        :return: tuple with padded image tensor and the N x 6 target tensor with the format: i, c, x, y, w, h
        """
        _, h, w = img.shape

        # Pad to square resolution
        img, pad = pad_to_square(img, self.padding_value)
        _, padded_h, padded_w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        boxes = None
        if len(anns) > 0:
            number_of_annotations = len(anns)
            boxes = torch.zeros((number_of_annotations, 6))

            for c, ann in enumerate(anns):
                bbox = ann["bbox"]

                xb, yb, wb, hb = bbox[0], bbox[1], bbox[2], bbox[3]

                # This adjustment is done when having bounding boxes outside the image boundaries
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
                boxes[c, 1] = ann["category_id"]
                boxes[c, 2] = (x1 + x2) / 2 / padded_w
                boxes[c, 3] = (y1 + y2) / 2 / padded_h
                boxes[c, 4] = (x2 - x1) * w_factor / padded_w
                boxes[c, 5] = (y2 - y1) * h_factor / padded_h
        boxes[:, 1] = torch.tensor(list(map(self.class_indices.index, boxes[:, 1].tolist())),
                                   device=boxes.device)
        return img, boxes

    def collate_fn(self, batch):
        img_paths, img_ids, imgs, targets = list(zip(*batch))
        for i, boxes in enumerate(targets):
            if boxes is not None:
                boxes[:, 0] = i
        targets = [boxes for boxes in targets if boxes is not None]
        targets = torch.cat(targets, 0)
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return img_paths, img_ids, imgs, targets


class COCODataset(_COCODataset):

    def __init__(self, root: str,
                 annotations_file: str,
                 img_size=416,
                 augment=True,
                 multiscale=True,
                 normalized_labels=True,
                 partition=None,
                 val_split=0.2,
                 seed=None,
                 padding_value=1,
                 views=None):
        """
        Dataset for coco-annotated data

        :param root: images root directory path
        :param img_size: rescaled image size (int)
        :param augment: boolean to apply data augmentation (horizontal flipping)
        :param multiscale: boolean to use multi-scale
        :param normalized_labels: indicates if labels in the annotation file are normalized
        :param partition: choose between train and val, None if there training and validation directories are separated.
            If None, then it means validation and training are in different folders and all data is taken
        :param val_split: split of the data for validation
        :param seed: random seed
        :param padding_value: padding value to add
        :param views: iterable with valid views. The view identifier must be at the end
        (and not including extension)
        """
        super().__init__(root, annotations_file, img_size, augment, multiscale, normalized_labels, partition,
                         val_split, seed, padding_value)
        self.wts = self._get_class_weights()
        self.views = views

    def _get_images(self):
        if self.views:
            return [{"id": int(img['id']),
                     "file_name": img['file_name']} for img in self._coco_file['images']
                    if img['file_name'].rsplit(".")[0].endswith(self.views)]
        else:
            return [{"id": int(img['id']),
                     "file_name": img['file_name']} for img in self._coco_file['images']]

    def _get_class_weights(self):
        wts = [0] * len(self.class_indices)
        for ann in self.anns:
            cat = ann['category_id']
            wts[self.class_indices.index(cat)] += 1
        wts = [sum(f for f in wts if f != w) / w for w in wts]
        return wts

    def __getitem__(self, index: int):

        img = self.imgs[index]
        anns = [a for a in self.anns if a["image_id"] == img["id"]]
        img_path = os.path.join(self.root, img["file_name"])
        img_id = img["id"]

        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        img, targets = self.anns_to_bounding_boxes(anns, img)
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horizontal_flip(img, targets)
        return img_path, img_id, img, targets

    def collate_fn(self, batch):
        img_paths, img_ids, imgs, targets = list(zip(*batch))
        for i, boxes in enumerate(targets):
            if boxes is not None:
                boxes[:, 0] = i
        targets = [boxes for boxes in targets if boxes is not None]
        targets = torch.cat(targets, 0)
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return img_paths, img_ids, imgs, targets


class MVCOCODataset(_COCODataset):
    def __init__(self, root: str,
                 annotations_file: str,
                 views=("A", "B"),
                 img_size=416,
                 multiscale=True,
                 normalized_labels=True,
                 partition=None,
                 val_split=0.2,
                 seed=None,
                 padding_value=0):
        super().__init__(root, annotations_file, img_size, False, multiscale, normalized_labels, partition,
                         val_split, seed, padding_value)
        self.views = views

    def _get_images(self):
        return self._coco_file["images"]

    def _get_annotations(self):
        return self._coco_file["annotations"]

    def _split(self):
        if self.partition is not None:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self.img_groups)

            total_groups = len(self.img_groups)
            total_partition_val = int(total_groups * self.val_split)

            self.img_groups = self.img_groups[:total_partition_val] if self.partition == "val" \
                else self.img_groups[total_partition_val:]

    def anns_to_bounding_boxes(self, anns, imgs):
        """
        Converts a region dict to bounding boxes
        :param anns: lists of anns dictionaries
        :param imgs: original images dict, per view
        :return: bounding boxes with format dict, and each view label includes a tensor as a value, with dims:
                number of annotations x 7 -> (batch_id, ann_id, label, x, y, w, h)
        """
        number_of_annotations = len(anns)
        n_views = len(self.views)
        boxes = torch.zeros((number_of_annotations, n_views, 6)) if number_of_annotations > 0 else None
        for i, v in enumerate(self.views):
            img = imgs[i]
            view_anns = [ann["views"][v] for ann in anns]
            for ann, view_ann in zip(anns, view_anns):
                view_ann["category_id"] = ann["category_id"]
            view_img, view_boxes = super().anns_to_bounding_boxes(view_anns, img)
            imgs[i] = view_img
            boxes[:, i, :] = view_boxes
        """
        imgs is a list of padded imgs
        boxes is a N x V x 6 tensor or None
        """
        return imgs, boxes

    def __getitem__(self, index: int):
        img_group = self.imgs[index]
        img_paths = tuple(os.path.join(self.root, img_group["views"][v]["file_name"]) for v in self.views)
        img_id = img_group["id"]
        anns = [ann for ann in self.anns if ann["image_group_id"] == img_id]
        valid_anns = copy.deepcopy(anns)

        imgs = []
        for img_path in img_paths:
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
            if len(img.shape) != 3:
                img = img.unsqueeze(0)
                img = img.expand((3, img.shape[1:]))
            imgs.append(img)

        imgs, targets = self.anns_to_bounding_boxes(valid_anns, imgs)
        """
        img_paths: tuple of strings, with the file names in order of self.views
        img_id: id of the image group
        imgs: list of C x W x H tensors
        targets: N x V x 6 tensor
        """
        return img_paths, img_id, imgs, targets

    def collate_fn(self, batch):
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        img_paths, img_ids, img_groups, targets = list(zip(*batch))
        for i, boxes in enumerate(targets):
            if boxes is not None:
                boxes[..., 0] = i
        targets = [boxes for boxes in targets if boxes is not None]
        if len(targets) > 0:
            targets = torch.cat(targets, 0)
        else:
            targets = torch.empty((0, 7))

        self.batch_count += 1

        imgs = torch.stack([torch.stack([resize(img, self.img_size) for img in img_group]) for img_group in img_groups])
        """
        img_paths is a list of tuples of strings
        img_id is a list of ids
        imgs is a B x V x C x W x H tensor
        targets is a BN x V x 6 tensor
        """
        return img_paths, img_ids, imgs, targets


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


class COCODatasetFromMV(COCODataset):

    def __init__(self, root: str,
                 annotations_file: str,
                 img_size=416,
                 augment=True,
                 multiscale=True,
                 normalized_labels=True,
                 partition=None,
                 val_split=0.2,
                 seed=None,
                 padding_value=0,
                 views=None):
        """
        Dataset for files with the structure

        - root
           |  - image1.jpg
           |  - image2.jpg
           |  - ...

        :param root: root directory path where images are
        :param img_size: rescaled image size
        :param augment: boolean indicating if data augmentation is going to be applied
        :param multiscale: indicates if multi-scale must be used
        :param normalized_labels: indicates if labels in the annotation file are normalized
        :param partition: choose between train and val, None if there training and validation directories are separated.
            If None, then it means validation and training are in different folders and all data is taken
        :param val_split: split of the data for validation
        :param seed: random seed
        """
        super(COCODatasetFromMV, self).__init__(root, annotations_file, img_size, augment, multiscale,
                                                normalized_labels, partition, val_split, seed, padding_value, views)

    def _get_images(self):
        imgs = []
        img_id = 1
        for im_group in self._coco_file["images"]:
            views_to_include = self.views if self.views else im_group["views"].keys()
            for v in views_to_include:
                im = {"id": img_id,
                      "file_name": im_group["views"][v]["file_name"]}
                imgs.append(im)
                img_id += 1
        return imgs

    def _get_annotations(self):
        anns = []
        for ann in self._coco_file["annotations"]:
            image_group_id = ann["image_group_id"]
            views_to_include = self.views if self.views else ann["views"].keys()
            for v in views_to_include:
                sv_ann = ann["views"][v]
                image_name = next(
                    im["views"][v]["file_name"] for im in self._coco_file["images"] if im["id"] == image_group_id)
                image_id = next(im["id"] for im in self.imgs if im["file_name"] == image_name)
                anns.append({"image_id": image_id, "category_id": ann["category_id"],
                             "bbox": sv_ann["bbox"]})
        return anns
