import datetime
import time

import visdom
from terminaltables import AsciiTable
from torch.utils.data import DataLoader

from yolov3.datasets import *
from yolov3.yolo import *
from yolov3.test import evaluate
from yolov3.utils.networks import weights_init_normal
from yolov3.utils.visualization import plot_lines

from yolov3.utils.parser import get_parser_from_arguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def estimate_reimaining_time(start_time, dataloader, current_batch):
    epoch_batches_left = len(dataloader) - (current_batch + 1)
    time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (current_batch + 1))
    return f"\n---- ETA {time_left}"


def get_dataset_args(parser):
    seed = random.randint(1, 100000)
    train_dataset_args = dict(
        root=parser.train["dir"],
        annotations_file=parser.train["annotation_file"],
        augment=parser.train["augment"],
        multiscale=parser.train["multiscale_training"],
        normalized_labels=parser.train["normalized"],
        img_size=parser.img_size
    )

    val_dataset_args = dict(
        root=parser.train["dir"],
        annotations_file=parser.train["annotation_file"],
        augment=False,
        multiscale=False,
        normalized_labels=parser.train["normalized"],
        img_size=parser.img_size
    )

    if parser.train["val_dir"] is None:
        train_dataset_args["partition"] = "train"
        train_dataset_args["val_split"] = parser.train["val_split"]
        train_dataset_args["seed"] = seed

        val_dataset_args["partition"] = "val"
        val_dataset_args["val_split"] = parser.train["val_split"]
        val_dataset_args["seed"] = seed
    else:
        val_dataset_args["root"] = parser.train["val_dir"]
        val_dataset_args["annotations_file"] = parser.train["val_annotation_file"]
    return train_dataset_args, val_dataset_args


def get_optimizer(parser, model):
    if parser.train["optimizer"]["type"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=parser.train["optimizer"]["lr"],
                                     weight_decay=parser.train["optimizer"]["decay"])
    else:
        momentum = parser.train["optimizer"]["momentum"]
        if momentum is None:
            momentum = 0.9
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=parser.train["optimizer"]["lr"],
                                    weight_decay=parser.train["optimizer"]["decay"],
                                    momentum=momentum)
    steps = parser.train["optimizer"]["scheduler_milestones"]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steps, parser.train["optimizer"]["gamma"]) \
        if steps else None
    return optimizer, scheduler


def train(parser):
    output_file = os.path.join(parser.weights_dir, parser.train["output_name"])
    viz = visdom.Visdom(server=parser.visdom["host"], port=parser.visdom["port"]) if parser.visdom["show"] else None

    train_dataset_args, val_dataset_args = get_dataset_args(parser)

    dataset = COCODataset(**train_dataset_args)
    eval_dataset = COCODataset(**val_dataset_args)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=parser.train["batch_size"],
        shuffle=True,
        num_workers=parser.workers,
        collate_fn=dataset.collate_fn,
    )
    # Initiate model
    model = YOLOv3(len(dataset.classes), anchors=parser.anchors).to(device)

    if parser.train["pretrained_weights"]:
        # noinspection PyTypeChecker
        if parser.train["pretrained_weights"].endswith((".pt", ".pth")):
            model.load_state_dict(torch.load(parser.train["pretrained_weights"]))
        else:
            model.load_yolov3_weights(parser.train["pretrained_weights"])
    else:
        model.apply(weights_init_normal)
    optimizer, scheduler = get_optimizer(parser, model)

    f_stat = open(os.path.join(parser.logs_dir, "stats.txt"), 'w')

    best_ap = []
    best_map = 0
    best_epoch = 0
    loss_win, ap_win = None, None
    train_losses = []
    val_losses = []
    class_aps = {class_name: [] for class_name in dataset.classes}
    class_aps["mAP"] = []

    for epoch in range(parser.train["epochs"]):
        model.train()
        start_time = time.time()
        train_loss = 0
        for batch_i, (img_paths, img_ids, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs, targets = imgs.to(device), targets.to(device)
            targets.requires_grad = False

            loss, outputs = model(imgs, targets)
            loss.backward()
            train_loss += loss.item()

            if not batches_done % parser.train["gradient_accumulations"]:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            log_str = f"\n---- [Epoch {epoch}/{parser.train['epochs']}, Batch {batch_i}/{len(dataloader)}] ----\n"
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            # Log metrics at each YOLO layer
            metric_table += model.get_metrics()
            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {float(loss.item())}"

            # Determine approximate time left for epoch
            log_str += estimate_reimaining_time(start_time, dataloader, batch_i)

            if batch_i % parser.train["train_metrics_print_interval"] == 0:
                print(log_str)
            model.seen += imgs.size(0)

        train_losses.append(train_loss / len(dataloader))
        if scheduler:
            scheduler.step()

        if epoch % parser.train["evaluation_interval"] == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            val_loss, precision, recall, AP, f1, ap_class = evaluate(
                eval_dataset,
                model,
                iou_thres=parser.train["iou_thres"],
                conf_thres=parser.train["conf_thres"],
                nms_thres=parser.train["nms_thres"],
                img_size=parser.img_size,
                workers=parser.workers,
                bs=parser.train["batch_size"]
            )
            mAP = AP.mean()

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, dataset.classes[c], "%.5f" % AP[i]]]
                class_aps[dataset.classes[c]].append(AP[i])
            class_aps["mAP"].append(mAP)
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            f_stat.write(f'---- epoch {epoch}\n{str(AsciiTable(ap_table).table)}\n---- mAP {str(AP.mean())}\n')

            if mAP >= best_map:
                best_ap = list("{0:0.4f}".format(i) for i in AP)
                best_map = mAP
                best_epoch = epoch
                torch.save(model.state_dict(), output_file)

            best_map_str = "{0:.4f}".format(best_map)

            print(f'<< Best Results|| Epoch {best_epoch} | Class {best_ap} | mAP {best_map_str} >>')
            f_stat.write(f'Best Results-->> Epoch {best_epoch} | Class {best_ap} | mAP {best_map_str}\n')

            val_losses.append(val_loss)
        if viz:
            x = np.array(list(range(epoch + 1)))
            loss_y = {
                "Train loss": train_losses,
                "Val loss": val_losses
            }
            loss_opts = dict(
                xtickmin=0,
                xtickmax=parser.train["epochs"],
                ytickmin=0,
                markers=True,
                xlabel="Epochs",
                ylabel="Loss",
                title='Train and Val Loss'
            )
            ap_opts = dict(
                xtickmin=0,
                xtickmax=parser.train["epochs"],
                ytickmin=0,
                ytickmax=1,
                markers=True,
                xlabel="Epochs",
                ylabel="Precision",
                title='Average Precision per class'
            )
            loss_win = plot_lines(x, loss_y, viz, opts=loss_opts, win=loss_win)
            ap_win = plot_lines(x, class_aps, viz, opts=ap_opts, win=ap_win)

        if epoch % parser.train["checkpoint_interval"] == 0:
            torch.save(model.state_dict(), os.path.join(parser.checkpoints_dir, f"yolov3_ckpt_{epoch}.pth"))

    f_stat.close()


if __name__ == '__main__':
    parser, opt = get_parser_from_arguments()
    train(parser)