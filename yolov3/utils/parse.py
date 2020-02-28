import yaml
import os


class ConfigParser:

    def __init__(self, config_filepath):
        """
        Auxiliar class for getting config data

        :param config_filepath: configuration file path
        """
        with open(config_filepath, 'r') as config:
            try:
                data_config = yaml.safe_load(config)
            except yaml.YAMLError as e:
                print(e)

        os.makedirs(data_config['output_folder'], exist_ok=True)
        self.db_name = os.path.join(data_config['output_folder'], data_config['experiment_name'])
        os.makedirs(self.db_name, exist_ok=True)

        # Getting model configuration
        anchors = data_config["model"]['anchors']
        self.anchors = [tuple(box) for box in anchors]

        # Getting training configuration
        self.logs_dir = os.path.join(self.db_name, 'logs')
        self.checkpoints_dir = os.path.join(self.db_name, "checkpoints")
        self.weights_dir = os.path.join(self.db_name, 'weights')
        self.inference_dir = os.path.join(self.db_name, "inference")

        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.inference_dir, exist_ok=True)

        self.workers = data_config["workers"]
        self.img_size = data_config["img_size"]

        self.train = {
            "dir": data_config["train"]["dir"],
            "annotation_file": data_config["train"]["annotation_file"],
            "val_dir": data_config["train"]["val_dir"],
            "val_annotation_file": data_config["train"]["val_annotation_file"],
            "val_split": float(data_config["train"]["val_split"]),
            "normalized": data_config["train"]["normalized"],
            "output_name": data_config["train"]["output_name"],
            "epochs": data_config["train"]["epochs"],
            "gradient_accumulations": data_config["train"]["gradient_accumulations"],
            "batch_size": data_config["train"]["batch_size"],
            "weights": data_config["train"]["weights"],
            "checkpoint_interval": data_config["train"]["checkpoint_interval"],
            "evaluation_interval": data_config["train"]["evaluation_interval"],
            "compute_map": data_config["train"]["compute_map"],
            "multiscale_training": data_config["train"]["multiscale_training"],
            "iou_thres": data_config["train"]["iou_thres"],
            "nms_thres": data_config["train"]["nms_thres"],
            "conf_thres": data_config["train"]["conf_thres"],
            "augment": data_config["train"]["augment"],
            "optimizer": {
                "type": data_config["train"]["optimizer"]["type"],
                "lr": float(data_config["train"]["optimizer"]["lr"]),
                "momentum": float(data_config["train"]["optimizer"]["momentum"]),
                "decay": float(data_config["train"]["optimizer"]["decay"]),
                "scheduler_milestones": data_config["train"]["optimizer"]["scheduler_milestones"],
                "gamma": float(data_config["train"]["optimizer"]["gamma"]),
            }
        }

        self.test = {
            "weights_file": data_config["test"]["weights_file"],
            "dir": data_config["test"]["dir"],
            "annotation_file": data_config["test"]["annotation_file"],
            "batch_size": data_config["train"]["batch_size"],
            "normalized": data_config["test"]["normalized"],
            "iou_thres": data_config["test"]["iou_thres"],
            "nms_thres": data_config["test"]["nms_thres"],
            "conf_thres": data_config["test"]["conf_thres"],
            "json_file_output": data_config["test"]["json_file_output"]
        }

        self.inference = {
            "weights_file": data_config["inference"]["weights_file"],
            "dir": data_config["inference"]["dir"],
            "output": data_config["inference"]["output"],
            "batch_size": data_config["train"]["batch_size"],
            "max_images": data_config["inference"]["max_images"],
            "classes": data_config["inference"]["classes"],
            "nms_thres": data_config["inference"]["nms_thres"],
            "conf_thres": data_config["inference"]["conf_thres"]
        }

        self.visdom = {
            "show": data_config["visdom"]["show"],
            "host": data_config["visdom"]["host"],
            "port": data_config["visdom"]["port"]
        }
