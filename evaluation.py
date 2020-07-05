import argparse
import json
import time
from typing import Union, List

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from matplotlib import pyplot as plt


class DetectionPerformanceEvaluation:

    def __init__(self, gt: Union[str, COCO], prediction: Union[List, str], params=None, th=0.5):
        """
        Helper class for performing coco evaluation
        :param gt: ground truth. It can be either a COCO object of a string with the path of the coco annotation file
        :param prediction: prediction output. It can be a list of dictionaries or a string with the file path
        :param params: COCO Params object
        :param th: confussion matrix IoU threshold
        """
        if isinstance(gt, str):
            gt = COCO(gt)

        prediction_coco = dict()
        if isinstance(prediction, str):
            print('loading detectron output annotations into memory...')
            tic = time.time()
            prediction = json.load(open(prediction, 'r'))  # Loading the json file as an array of dicts
            assert type(prediction) == list, 'annotation file format {} not supported'.format(
                type(prediction))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))

        for i, p in enumerate(prediction):
            p['id'] = i
            p['segmentation'] = []
            p['area'] = p['bbox'][2] * p['bbox'][3]
            if "file_name" in p.keys():
                p["image_id"] = next(j for j, im in gt.imgs.items() if im["file_name"] == p["file_name"])
        # Adding these lines I give the detection file the xray format
        prediction_coco["annotations"] = prediction
        prediction_coco["images"] = gt.dataset["images"]
        prediction_coco["categories"] = gt.dataset["categories"]

        # COCO object instantiation
        prediction = COCO()
        prediction.dataset = prediction_coco
        prediction.createIndex()

        self.ground_truth = gt
        self.prediction = prediction
        self.eval = COCOeval(gt, prediction, iouType='bbox')
        self.params = self.eval.params
        self._imgIds = gt.getImgIds()
        self._catIds = gt.getCatIds()
        self.th = th
        if params:
            self.params = params
            self.eval.params = params
            self.eval.params.imgIds = sorted(self._imgIds)
            self.eval.params.catIds = sorted(self._catIds)

    def _build_no_cat_params(self):
        params = Params(iouType='bbox')
        params.maxDets = [500]
        params.areaRng = [[0 ** 2, 1e5 ** 2]]
        params.areaRngLbl = ['all']
        params.useCats = 0
        params.iouThrs = [self.th]
        return params

    def build_confussion_matrix(self, out_image_filename=None):
        params = self._build_no_cat_params()
        self.eval.params = params
        self.eval.params.imgIds = sorted(self._imgIds)
        self.eval.params.catIds = sorted(self._catIds)
        self.eval.evaluate()

        ann_true = []
        ann_pred = []

        nones = 0
        for evalImg, ((k, _), ious) in zip(self.eval.evalImgs, self.eval.ious.items()):
            if evalImg:
                ann_true += evalImg['gtIds']
                if len(ious) > 0:
                    valid_ious = (ious >= self.th) * ious
                    matches = valid_ious.argmax(0)
                    matches[valid_ious.max(0) == 0] = -1
                    ann_pred += [evalImg['dtIds'][match] if match > -1 else -1 for match in matches]
                else:
                    ann_pred += ([-1] * len(evalImg['gtIds']))
            else:
                nones += 1
        print(f"Nones: {nones}")

        y_true = [ann['category_id'] for ann in self.ground_truth.loadAnns(ann_true)]
        y_pred = [-1 if ann == -1 else self.prediction.loadAnns(ann)[0]['category_id'] for ann in ann_pred]
        y_true = [y + 1 for y in y_true]
        y_pred = [y + 1 for y in y_pred]
        cats = ['background'] + [cat['name'] for _, cat in self.ground_truth.cats.items()]
        cnf_mtx = confusion_matrix(y_true, y_pred, normalize='true')
        print(cnf_mtx)
        cnf_mtx_display = ConfusionMatrixDisplay(cnf_mtx, cats)
        _, ax = plt.subplots(figsize=(15, 15))
        cnf_mtx_display.plot(ax=ax, values_format='.3f')
        if out_image_filename is not None:
            cnf_mtx_display.figure_.savefig(out_image_filename)
        print(classification_report(y_true, y_pred, target_names=cats))
        pass

    def run_coco_metrics(self):
        self.eval.params = self.params
        self.eval.params.imgIds = sorted(self._imgIds)
        self.eval.params.catIds = sorted(self._catIds)
        self.eval.evaluate()
        self.eval.accumulate()
        self.eval.summarize()
        return self.eval.stats


def build_params():
    params = Params(iouType='bbox')
    params.maxDets = [1, 100, 500]
    params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    params.areaRngLbl = ['all', 'small', 'medium', 'large']
    params.useCats = 1
    return params


def evaluate(gt, preds, output_image=None):
    confusion_matrix_iou_threshold = 0.5

    params = build_params()  # Params for COCO metrics
    performance_evaluation = DetectionPerformanceEvaluation(gt, preds, params=params,
                                                            th=confusion_matrix_iou_threshold)
    if output_image:
        performance_evaluation.build_confussion_matrix(output_image)
    return performance_evaluation.run_coco_metrics()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--gt_path", help="Path to ground truth coco annotation file")
    args.add_argument("--pred_path", help="Path to predictions path")
    args.add_argument("--confmat_output", help="Confussion Matrix output file", default=None)
    opts = args.parse_args()
    evaluate(opts.gt_path, opts.pred_path, opts.confmat_output)
