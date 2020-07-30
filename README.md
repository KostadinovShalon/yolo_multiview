# Multi-view Epipolar Filtering

An implementation of epipolar constraints at test time for multi-view object detection. A summary of the changes
is described in the [changelog](CHANGELOG.md). Current version is 1.0.0.

## Dataset format
For this project, a new annotations file format based on the COCO format is used. The annotations 
file must be a json file with the following fields

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

To convert a coco annotation file to the multi-view annotation format described, the script `coco_to_mvcoco.py` is
provided.

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
      --gt_path GT_PATH     YAML Config file path
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
