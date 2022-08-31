# Multi-view Epipolar Filtering

Implementation of  the paper [Multi-view Object Detection Using Epipolar Constraints within Cluttered X-ray Security Imagery](https://breckon.org/toby/publications/papers/isaac20multiview.pdf) (Isaac-Medina et al., 2020).
In this paper, we use a filtering technique based on epipolar constraints for multi-view object detection.
A requirements file is also provided.

## Train
Our paper uses a YOLOv3 detector but any detector that give a score to a bounding box could be used. To train the 
detector, use
    
    python train.py [-h] [--gt_path GT_PATH] [--type {sv,mv}]
               [--config_file CONFIG_FILE] [--output_folder OUTPUT_FOLDER]
               [--experiment_name EXPERIMENT_NAME] [--model_path MODEL_PATH]
               
    Obligatory arguments:
    --config_file CONFIG_FILE
                            YAML Config file path
               
    optional arguments:
      -h, --help            show this help message and exit
      --output_folder OUTPUT_FOLDER
                            Output folder where training results are saved
      --experiment_name EXPERIMENT_NAME
                            Name of the experiment


## Test

All parameters are defined in the config file. Check the example [config file](config.yaml). For multi-view testing, run the following command

    python test.py [-h] [--gt_path GT_PATH] [--type {sv,mv}]
               [--config_file CONFIG_FILE] [--output_folder OUTPUT_FOLDER]
               [--experiment_name EXPERIMENT_NAME] [--model_path MODEL_PATH]
               
    Obligatory arguments:
    --config_file CONFIG_FILE
                            YAML Config file path
               
    optional arguments:
      -h, --help            show this help message and exit
      --gt_path GT_PATH     Path to the COCO (single-view) ground truth annotations file
      --type {sv,mv}        Testing type: mv (default) for multi-view and sv for
                            single-view
      --output_folder OUTPUT_FOLDER
                            Output folder where experiments are saved
      --experiment_name EXPERIMENT_NAME
                            Name of the experiment
      --model_path MODEL_PATH
                            Model path for inference


## Inference

For multi-view inference, run the following command

    python inference.py [-h] [--gt_path GT_PATH] [--type {sv,mv}]
                    [--config_file CONFIG_FILE]
                    [--output_folder OUTPUT_FOLDER]
                    [--experiment_name EXPERIMENT_NAME]
                    [--model_path MODEL_PATH]
                    
    Obligatory arguments:
    --config_file CONFIG_FILE
                            YAML Config file path

    Optional arguments:
      -h, --help            show this help message and exit
      --gt_path GT_PATH     YAML Config file path
      --type {sv,mv}        Testing type: mv (default_ for multi-view and sv for
                            single-view
      --output_folder OUTPUT_FOLDER
                            Output folder where experiments are saved
      --experiment_name EXPERIMENT_NAME
                            Name of the experiment
      --model_path MODEL_PATH


## Dataset format
For this project, a new annotations file based on the COCO annotations is used. The annotations 
file must be a json file with the following format

    {
        "info"              : info,
        "images"            : [image_group],
        "annotations"       : [annotation],
        "licenses"          : [license],
        "categories"        : [category]
    }
    
    info{
        "year"              : int,
        "version"           : str,
        "description"       : str,
        "contributor"       : str,
        "url"               : str,
        "date_created"      : str
    }
    
    license{
        "id"                : int,
        "name"              : str,
        "url"               : str
    }
    
    image_group{
        "id"                : int,
        "tag"               : str,
        "license"           : int,
        "views"             : {view}
    }
    
    view --> view_name: {
        "file_name"         : str,
        "width"             : int,
        "height"            : int
    }
    
    annotation{
        "id"                : int,
        "image_group_id"    : int,
        "category_id"       : int,
        "views"             : {view_annotation}
    }
    
    view_annotation --> view_name: {
        "segmentation"      : [polygons or RLE],
        "iscrowd"           : int,
        "bbox"              : [x, y, w, h],
        "area"              : float
    }
    
    category{
        "id"                : int,
        "name"              : str,
        "supercategory"     : str
    }

In the `bbox` field of a view_annotation, x and y are the top left coordinates of the bounding box and
w and h are the width and height. The key `view_name` refers to the view identifier.

To convert a coco annotation file to the multi-view annotation format described, use

`python colo_to_mvcoco --json_file <PATH TO THE COCO ANNOTATIONS> --separator <file names separator> --output <PATH TO OUTPUT FILE>`
