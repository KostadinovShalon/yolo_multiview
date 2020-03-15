from datasets import MVCOCODataset
from torch.utils.data import DataLoader
import torch
import datetime
import time
from matplotlib import pyplot as plt
from yolov3.mvyolo import MVYOLOv3
import random
from yolov3.utils.networks import weights_init_normal
import os
import visdom
from test import evaluate
from terminaltables import AsciiTable

root_dir = '/home/brian/Documents/datasets/smith_coco_no_electronics/train/images'
anns_file = '/home/brian/Documents/datasets/smith_coco_no_electronics/train/coco_annotations_no_electronics_mv.json'
pretrained = '/home/brian/Documents/Projects/yolov3/pretrained_weights/yolov3-openimages.weights'
outfile_name = 'test2.pth'
stats_name = 'stats2.txt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0001
epochs = 100
evaluation_interval = 1
checkpoint_interval = 5
gradient_accumulation = 4
bs = 2


def estimate_reimaining_time(start_time, dataloader, current_batch):
    epoch_batches_left = len(dataloader) - (current_batch + 1)
    time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (current_batch + 1))
    return f"\n---- ETA {time_left}"


def plot(train_loss, ap, title, vis, win=None):
    f = plt.figure(figsize=(16, 8))
    f.suptitle(title)
    ax = f.add_subplot(1, 2, 1)
    ax.plot(train_loss)
    ax.set_title('Train Loss.')
    ax.set_xlabel('Epoch')

    ax = f.add_subplot(1, 2, 2)
    ax.plot(ap)
    ax.set_title('Test AP.')
    ax.set_xlabel('Epoch')

    if win is None:
        win = vis.matplot(f)
    else:
        vis.matplot(f, win=win)

    plt.close(f)

    return win


def plot2(data, title, vis, win=None):
    n_plots = len(data.keys())
    f = plt.figure(figsize=(8 * n_plots, 8))
    f.suptitle(title)
    for i, (k, d) in enumerate(data.items()):
        ax = f.add_subplot(1, n_plots, i + 1)
        ax.plot(d)
        ax.set_title(k)
        ax.set_xlabel('Epoch')

    if win is None:
        win = vis.matplot(f)
    else:
        vis.matplot(f, win=win)

    plt.close(f)

    return win


# noinspection PyUnresolvedReferences
def train():

    seed = random.randint(1, 100000)
    # Get dataloader
    train_dataset_args = dict(
        root=root_dir,
        annotations_file=anns_file,
        normalized_labels=False,
        seed=seed
    )

    val_dataset_args = dict(
        root=root_dir,
        annotations_file=anns_file,
        multiscale=False,
        normalized_labels=False,
        seed=seed
    )

    train_dataset_args["partition"] = "train"
    train_dataset_args["val_split"] = 0.2
    train_dataset_args["seed"] = seed

    val_dataset_args["partition"] = "val"
    val_dataset_args["val_split"] = 0.2
    val_dataset_args["seed"] = seed

    dataset = MVCOCODataset(**train_dataset_args)

    eval_dataset = MVCOCODataset(**val_dataset_args)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )

    # Initiate model
    model = MVYOLOv3(len(dataset.classes)).to(device)
    model.apply(weights_init_normal)

    if pretrained:
        # noinspection PyTypeChecker
        if pretrained.endswith(".pth"):
            model.load_state_dict(torch.load(pretrained))
        else:
            model.load_yolov3_weights(pretrained)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=5e-4)

    steps = [40, 60]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steps, 0.1)

    output_file = os.path.join("weights", outfile_name)

    viz = visdom.Visdom()

    with open(os.path.join("logs", stats_name), 'w') as f_stat:

        best_ap = []
        best_map = 0
        best_epoch = 0
        win = None
        losses = []
        aps = []
        yolo_losses = []
        projection_losses = []
        contrastive_losses = []

        for epoch in range(epochs):
            model.train()
            start_time = time.time()
            acc_loss = 0
            acc_yolo_loss = 0
            acc_proj_loss = 0
            acc_contr_loss = 0
            for batch_i, (imgs, targets) in enumerate(dataloader):
                batches_done = len(dataloader) * epoch + batch_i
                num_imgs = 0
                for k in imgs.keys():
                    num_imgs = imgs[k].size(0)
                    imgs[k] = imgs[k].to(device)
                    targets[k] = targets[k].to(device)
                    targets[k].requires_grad = False

                loss, outputs = model(imgs, targets)
                yolo_loss, projected_loss, contrastive_loss = loss
                loss = sum(loss)
                loss.backward()

                acc_loss += loss.item()
                try:
                    acc_yolo_loss += yolo_loss.item()
                    acc_proj_loss += projected_loss.item()
                    acc_contr_loss += contrastive_loss.item()
                except AttributeError:
                    pass

                if not batches_done % gradient_accumulation:
                    # Accumulates gradient before each step
                    optimizer.step()
                    optimizer.zero_grad()

                log_str = f"\n---- [Epoch {epoch}/{epochs}, Batch {batch_i}/{len(dataloader)}] ----\n"
                log_str += f"\nTotal loss {float(loss.item())}"

                # Determine approximate time left for epoch
                log_str += estimate_reimaining_time(start_time, dataloader, batch_i)

                print(log_str)

                model.seen += num_imgs

            losses.append(acc_loss / (len(dataloader) * bs))
            yolo_losses.append(acc_yolo_loss / (len(dataloader) * bs))
            projection_losses.append(acc_proj_loss / (len(dataloader) * bs))
            contrastive_losses.append(acc_contr_loss / (len(dataloader) * bs))

            if scheduler:
                scheduler.step()

            if epoch % evaluation_interval == 0:
                print("\n---- Evaluating Model ----")
                # Evaluate the model on the validation set
                precision, recall, AP, f1, ap_class = evaluate(
                    eval_dataset,
                    model,
                    iou_thres=0.5,
                    conf_thres=0.5,
                    nms_thres=0.4,
                    img_size=416,
                    workers=4,
                    bs=2
                )

                # Print class APs and mAP
                ap_table = [["Index", "Class name", "AP"]]
                try:
                    for i, c in enumerate(ap_class):
                        ap_table += [[c, dataset.get_cat_by_positional_id(c), "%.5f" % AP[i]]]
                    print(AsciiTable(ap_table).table)
                    print(f"---- mAP {AP.mean()}")
                    f_stat.write(f'---- epoch {epoch}\n{str(AsciiTable(ap_table).table)}\n---- mAP {str(AP.mean())}\n')

                    if AP.mean() >= best_map:
                        best_ap = list("{0:0.4f}".format(i) for i in AP)
                        best_map = AP.mean()
                        best_epoch = epoch
                        torch.save(model.state_dict(), output_file)
                    aps.append(AP.mean())
                except TypeError:
                    f_stat.write(f'---- epoch {epoch}\n---- mAP {0}\n')
                    aps.append(0)

                best_map_str = "{0:.4f}".format(best_map)

                print(f'<< Best Results|| Epoch {best_epoch} | Class {best_ap} | mAP {best_map_str} >>')
                f_stat.write(f'Best Results-->> Epoch {best_epoch} | Class {best_ap} | mAP {best_map_str}\n')

            if viz:
                # win = plot(losses, aps, 'YOLOv3', viz, win)
                data = {
                    'mAP': aps,
                    'Total Loss': losses,
                    'YOLO Loss': yolo_losses,
                    'Projection Loss': projection_losses,
                    'Contrastive Loss': contrastive_losses
                }
                win = plot2(data, 'MVYOLOv3', viz, win)

            if epoch % checkpoint_interval == 0:
                torch.save(model.state_dict(), os.path.join("checkpoints", f"ckpt_{epoch}_{outfile_name}.pth"))


if __name__ == '__main__':
    train()
