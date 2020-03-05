from datasets import MVCOCODataset
from torch.utils.data import DataLoader
root_dir = '/home/brian/Documents/datasets/smith_coco_no_electronics/train/images'
anns_file = '/home/brian/Documents/datasets/smith_coco_no_electronics/train/coco_annotations_no_electronics_mv.json'
dataset = MVCOCODataset(root_dir, anns_file, multiscale=True, normalized_labels=False)

dataloader = DataLoader(dataset, 8, False, num_workers=4, collate_fn=dataset.collate_fn)

for data in dataloader:
    print(data)