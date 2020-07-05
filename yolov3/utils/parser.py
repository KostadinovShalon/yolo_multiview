import argparse

import yaml
import os
from datetime import datetime


def get_parser_from_arguments():
    args = argparse.ArgumentParser()
    args.add_argument("--config_file", help="YAML Config file path")
    args.add_argument("--output_folder", help="Output folder where experiments are saved")
    args.add_argument("--experiment_name", help="Name of the experiment")

    args.add_argument("--model_path", help="Model path for inference")
    opt = args.parse_args()

    parser = ConfigParser(opt.config_file, create_folders=False)
    if opt.output_folder:
        parser.output_folder = opt.output_folder
    if opt.experiment_name:
        parser.experiment_name = opt.experiment_name
    if opt.model_path and parser.inference:
        parser.inference["weights_file"] = opt.model_path
    parser.create_folders()
    return parser


class ConfigParser:

    def __init__(self, config_filepath, create_folders=True):
        """
        Auxiliar class for getting config data

        :param config_filepath: configuration file path
        """
        with open(config_filepath, 'r') as config:
            try:
                self._config = yaml.safe_load(config)
            except yaml.YAMLError as e:
                print(e)

        self.output_folder = self.get_or_default('output_folder', default='experiments')
        self.experiment_name = self.get_or_default('experiment_name',
                                                   f'experiment_{datetime.now().strftime("%y%m%d_%H%M%S")}')
        self.db_name = os.path.join(self.output_folder, self.experiment_name)

        # Getting model configuration
        anchors = self._config["model"]['anchors']
        self.anchors = [tuple(box) for box in anchors]

        # Getting training configuration
        self.logs_dir = os.path.join(self.db_name, 'logs')
        self.checkpoints_dir = os.path.join(self.db_name, "checkpoints")
        self.weights_dir = os.path.join(self.db_name, 'weights')
        self.inference_dir = os.path.join(self.db_name, "inference")

        if create_folders:
            self.create_folders()

        self.workers = self.get_or_default("workers", 4)
        self.img_size = self.get_or_default("img_size", 544)

        self.train = self.get_or_default("train")
        self.test = self.get_or_default("test")
        self.inference = self.get_or_default("inference")
        self.visdom = self.get_or_default("visdom")

    def get_or_default(self, key, default=None):
        return (self._config[key] if key in self._config else default) if self._config else default

    def create_folders(self):
        self.db_name = os.path.join(self.output_folder, self.experiment_name)
        self.logs_dir = os.path.join(self.db_name, 'logs')
        self.checkpoints_dir = os.path.join(self.db_name, "checkpoints")
        self.weights_dir = os.path.join(self.db_name, 'weights')
        self.inference_dir = os.path.join(self.db_name, "inference")

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.db_name, exist_ok=True)

        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.inference_dir, exist_ok=True)