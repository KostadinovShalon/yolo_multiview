import json
from itertools import permutations

import numpy as np
import torch
import torch.utils.data

import evaluation
from yolov3.datasets import COCODataset, MVCOCODataset, COCODatasetFromMV
from yolov3.epipolar_geometry import compute_fundamental_matrix
from yolov3.test import evaluate_singleview, evaluate_multiview
from yolov3.utils.parser import get_parser_from_arguments
from yolov3.yolo import YOLOv3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_single_view(parser, valid_views=None):
    print("Runnning Test")
    dataset = COCODataset(parser.test["dir"],
                          annotations_file=parser.test["annotation_file"],
                          augment=False,
                          multiscale=False,
                          normalized_labels=parser.test["normalized"],
                          views=valid_views,
                          img_size=parser.img_size,
                          padding_value=1)
    # Initiate model
    model = YOLOv3(len(dataset.classes), anchors=parser.anchors).to(device)

    _, precision, recall, AP, f1, ap_class, detections = evaluate_singleview(
        dataset,
        model,
        parser.test["iou_thres"],
        parser.test["conf_thres"],
        parser.test["nms_thres"],
        parser.img_size,
        parser.workers,
        parser.test["weights_file"],
        parser.test["batch_size"],
        return_detections=True
    )

    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({dataset.class_indices[c]}) - AP: {AP[i]} - precision: {precision[i]}"
              f" - recall {recall[i]}")

    print(f"mAP: {AP.mean()}")

    return detections

    # json_file_name = os.path.join(parser.db_name, parser.test["json_file_output"])
    #
    # with open(json_file_name, 'w') as f:
    #     json.dump(detections, f)


def test_multi_view(parser, views=("A", "B"), weak_th=0.01, score_th=0.3, f_matrices=None):
    print("Getting Fundamental Matrices")
    with open(parser.train["annotation_file"], 'r') as f:
        coco = json.load(f)
    perms = permutations(views, 2)

    print("Runnning Test")
    dataset = MVCOCODataset(parser.test["dir"],
                            views=views,
                            annotations_file=parser.test["annotation_file"],
                            multiscale=False,
                            normalized_labels=parser.test["normalized"],
                            img_size=parser.img_size,
                            padding_value=1)

    if f_matrices is None:
        f_matrices = {tuple([*perm, c]): compute_fundamental_matrix(coco, *perm, img_size=parser.img_size, class_id=c)
                      for perm in perms for c in dataset.class_indices}

    # Initiate model
    model = YOLOv3(len(dataset.classes), anchors=parser.anchors).to(device)

    if parser.test["weights_file"] is not None:
        if parser.test["weights_file"].endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(parser.test["weights_file"])
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(parser.test["weights_file"]))
    model.eval()
    _, precision, recall, AP, f1, ap_class, detections = evaluate_multiview(dataset,
                                                                            model,
                                                                            parser.test["iou_thres"],
                                                                            parser.test["conf_thres"],
                                                                            weak_th,
                                                                            parser.test["nms_thres"],
                                                                            parser.img_size,
                                                                            score_th,
                                                                            parser.workers,
                                                                            f_matrices,
                                                                            views,
                                                                            bs=parser.test["batch_size"],
                                                                            return_detections=True)

    # for i, c in enumerate(ap_class):
    #     print(f"+ Class '{c}' ({dataset.get_cat_by_positional_id(c)}) - AP: {AP[i]} - precision: {precision[i]}"
    #           f" - recall {recall[i]}")
    #
    # print(f"mAP: {AP.mean()}")

    return detections

    # json_file_name = os.path.join(parser.db_name, parser.test["json_file_output"])
    #
    # with open(json_file_name, 'w') as f:
    #     json.dump(detections, f)


def test_single_view2(parser, views=("A", "B")):
    # with open(parser.train["annotation_file"], 'r') as f:
    #     coco = json.load(f)
    # perms = permutations(views, 2)
    # f_matrices = {perm: compute_fundamental_matrix(coco, *perm) for perm in perms}

    print("Runnning Test")
    dataset = COCODatasetFromMV(parser.test["dir"],
                                annotations_file=parser.test["annotation_file"],
                                multiscale=False,
                                augment=False,
                                normalized_labels=parser.test["normalized"],
                                img_size=parser.img_size,
                                views=views,
                                padding_value=1)

    # Initiate model
    model = YOLOv3(len(dataset.classes), anchors=parser.anchors).to(device)

    if parser.test["weights_file"] is not None:
        if parser.test["weights_file"].endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(parser.test["weights_file"])
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(parser.test["weights_file"]))
    model.eval()

    _, precision, recall, AP, f1, ap_class, detections = evaluate_singleview(
        dataset,
        model,
        parser.test["iou_thres"],
        parser.test["conf_thres"],
        parser.test["nms_thres"],
        parser.img_size,
        parser.workers,
        parser.test["weights_file"],
        parser.test["batch_size"],
        return_detections=True
    )

    # for i, c in enumerate(ap_class):
    #     print(f"+ Class '{c}' ({dataset.get_cat_by_positional_id(c)}) - AP: {AP[i]} - precision: {precision[i]}"
    #           f" - recall {recall[i]}")
    #
    # print(f"mAP: {AP.mean()}")

    return detections

    # json_file_name = os.path.join(parser.db_name, parser.test["json_file_output"])
    #
    # with open(json_file_name, 'w') as f:
    #     json.dump(detections, f)


def get_matrices():
    return {('A', 'B', 1): (np.array([[-3.75529011e-08, -1.64293354e-07, -9.74453879e-03],
                                      [-3.74756394e-07, 1.34064598e-09, 1.57336122e-04],
                                      [9.93107299e-03, 3.36792723e-05, -6.13736306e-02]]), 0.03205558711190775,
                            1.8154441977086766),
            ('A', 'B', 2): (np.array([[3.30296215e-06, 9.14764774e-06, 1.74393296e-02],
                                      [-6.15558318e-06, -8.88062077e-09, 1.79308157e-03],
                                      [-2.03338030e-02, -2.69037439e-03, 5.67354058e-01]]), 0.11346449472459673,
                            2.0260713907656167),
            ('A', 'B', 3): (np.array([[3.52565778e-07, 6.96864463e-08, 1.01147058e-02],
                                      [9.67420040e-07, 1.62526448e-08, -1.87062195e-04],
                                      [-1.06183560e-02, -1.90382070e-04, 1.37796564e-01]]), -0.13677395228138325,
                            2.921226442890686),
            ('A', 'B', 4): (np.array([[-4.94489373e-07, 8.03437517e-07, -1.10325733e-02],
                                      [-2.00188573e-06, 5.51333974e-09, 3.37829270e-04],
                                      [1.16955226e-02, -1.67043254e-04, -1.05025273e-01]]), 0.12901145290341248,
                            3.2023637017144244),
            ('A', 'C', 1): (np.array([[5.26064024e-08, 5.70611245e-07, 1.01454353e-02],
                                      [-2.25611831e-07, -9.33302786e-10, 6.25342258e-05],
                                      [-1.02554035e-02, -1.99708109e-04, 4.09666266e-02]]), 0.275348956264891,
                            3.1666062577077474),
            ('A', 'C', 2): (np.array([[-9.07641204e-06, -3.99993373e-06, 2.42866321e-02],
                                      [7.24449237e-06, -3.95906170e-08, -2.43765429e-03],
                                      [-2.03082464e-02, 1.47741242e-03, -3.31080483e-01]]), 0.3480936328462325,
                            2.47390843273624),
            ('A', 'C', 3): (np.array([[3.57357616e-07, 4.84123336e-07, -1.10649227e-02],
                                      [-1.26730910e-06, 1.09421595e-08, 4.30795627e-04],
                                      [1.12670319e-02, -2.63971217e-04, -8.58912985e-02]]), -0.438201608570547,
                            4.160244995603259),
            ('A', 'C', 4): (np.array([[1.04760042e-06, -1.36787293e-06, 1.24688995e-02],
                                      [-3.48711424e-06, 4.61248018e-09, 7.56825870e-04],
                                      [-1.15425672e-02, 3.09475504e-04, -2.28394506e-01]]), -0.06972104123650755,
                            3.568695308089554),
            ('A', 'D', 1): (np.array([[3.27273286e-07, 2.20368492e-06, 1.01036740e-02],
                                      [-3.23478204e-06, -1.28156788e-08, 9.87329136e-04],
                                      [-1.00379860e-02, -7.07620661e-04, -3.03051879e-02]]), -0.521881975697132,
                            3.728220723889957),
            ('A', 'D', 2): (np.array([[1.28812955e-06, -6.92865718e-06, 2.42322504e-02],
                                      [1.65189311e-06, -1.37223272e-07, -1.01614102e-04],
                                      [-2.38118725e-02, 2.51191620e-03, -4.37838976e-01]]), -0.3967011185300554,
                            2.9853421354124885),
            ('A', 'D', 3): (np.array([[2.17233033e-06, -2.39818645e-06, -1.16049531e-02],
                                      [4.86861969e-06, 1.73865356e-08, -1.67939656e-03],
                                      [9.36391901e-03, 8.57716660e-04, 4.89118910e-01]]), 1.5639636078216945,
                            4.660567913821218),
            ('A', 'D', 4): (np.array([[-2.51122446e-06, -1.03560182e-05, 1.45297548e-02],
                                      [9.66597899e-06, -3.29739677e-08, -3.79943830e-03],
                                      [-1.25751451e-02, 4.12778390e-03, -3.94473340e-01]]), -1.1746109836991125,
                            2.8628894998687557),
            ('B', 'A', 1): (np.array([[2.64432625e-07, 7.75583153e-07, 9.62187119e-03],
                                      [-3.64206366e-07, -9.60607376e-10, 7.99273750e-05],
                                      [-9.88850299e-03, -1.95911083e-04, 4.85735622e-02]]), -0.31898143496560755,
                            1.8083630202494199),
            ('B', 'A', 2): (np.array([[7.17425914e-07, -1.44822503e-06, 2.09903332e-02],
                                      [5.93184788e-06, 4.39579037e-08, -1.73245357e-03],
                                      [-2.25413603e-02, 2.54604568e-04, 4.33710470e-01]]), -0.2050771043244269,
                            1.943396593969977),
            ('B', 'A', 3): (np.array([[-2.85666898e-07, 3.01813733e-07, 1.04511486e-02],
                                      [5.73174385e-08, 4.06887307e-09, 1.13963014e-04],
                                      [-1.04007717e-02, -1.34420667e-04, -1.91107865e-02]]), -0.27176686220405477,
                            2.903384017409568),
            ('B', 'A', 4): (np.array([[4.02725932e-06, 5.55994671e-06, 8.61881294e-03],
                                      [-5.97232473e-06, -4.51744416e-08, 2.87917841e-03],
                                      [-1.10660907e-02, -2.70446650e-03, 3.29638846e-01]]), 0.4104544268710088,
                            3.750871432034983),
            ('B', 'C', 1): (np.array([[9.75477889e-08, -4.47032114e-07, 1.07146613e-02],
                                      [8.57049634e-07, 1.13526281e-09, -3.40172708e-04],
                                      [-1.09574221e-02, 1.62832772e-04, 8.42631376e-02]]), -0.13373439100828727,
                            3.1924407161650605),
            ('B', 'C', 2): (np.array([[1.30114962e-06, -1.15348034e-05, 2.23272974e-02],
                                      [-2.08378741e-06, 8.35108646e-08, 4.56970711e-04],
                                      [-1.92935924e-02, 3.40211733e-03, -9.46003057e-01]]), 0.18155480815599456,
                            2.1868336531687698),
            ('B', 'C', 3): (np.array([[-1.63839192e-07, 1.26248265e-06, -1.10477182e-02],
                                      [-1.02022471e-07, -1.32460169e-08, 1.57030454e-04],
                                      [1.08911136e-02, -4.95729880e-04, 4.68922320e-02]]), -0.3483379604355908,
                            4.145804793810657),
            ('B', 'C', 4): (np.array([[4.09132751e-06, -2.87127507e-07, 1.15761681e-02],
                                      [3.86178454e-06, -9.55595927e-08, -5.45396425e-04],
                                      [-1.54124932e-02, 4.08106537e-04, 4.29291718e-01]]), -0.7344546992744537,
                            3.3840237272671945),
            ('B', 'D', 1): (np.array([[1.39140686e-07, -3.27879807e-06, -9.27326225e-03],
                                      [2.60580928e-06, 3.61363997e-08, -6.58287448e-04],
                                      [9.29988580e-03, 9.45744363e-04, -4.94251723e-02]]), 1.3172958063401152,
                            3.787840280905243),
            ('B', 'D', 2): (np.array([[-2.16472170e-05, 3.38850343e-05, 1.93633445e-02],
                                      [-2.73892142e-05, 4.69305740e-08, 8.66398030e-03],
                                      [-8.18328888e-03, -1.06627546e-02, -1.35927295e+00]]), -1.3022801110786415,
                            3.631925391556244),
            ('B', 'D', 3): (np.array([[7.68289444e-07, -3.01771722e-07, -1.12555369e-02],
                                      [-1.97721265e-07, -6.84165708e-09, -4.43811592e-05],
                                      [1.08448669e-02, 3.05964555e-04, 2.25465597e-02]]), 1.4399732854092697,
                            4.277485300518875),
            ('B', 'D', 4): (np.array([[6.59840905e-07, -1.58656655e-06, 1.12823519e-02],
                                      [-8.68603658e-07, -7.26694714e-08, 6.58671296e-04],
                                      [-1.09191852e-02, 2.71564673e-04, -2.25062645e-01]]), -0.5469439773016519,
                            2.966438449797102),
            ('C', 'A', 1): (np.array([[-2.21389885e-07, -3.39805454e-07, 1.08401460e-02],
                                      [6.74995142e-07, 2.50850802e-09, -1.89645022e-04],
                                      [-1.07190370e-02, 5.49897050e-05, -3.37057522e-03]]), -0.16258652307733729,
                            3.211435403983224),
            ('C', 'A', 2): (np.array([[3.55511919e-06, 2.37055964e-06, 1.80375321e-02],
                                      [1.59115987e-05, -2.69443914e-07, -4.79702460e-03],
                                      [-2.55998782e-02, -3.15618895e-04, 1.82865136e+00]]), -0.34762878190752555,
                            2.474094225867839),
            ('C', 'A', 3): (np.array([[-1.67638251e-07, -5.70725495e-07, -1.06034933e-02],
                                      [-9.22070404e-07, 9.58186644e-09, 4.27845147e-04],
                                      [1.11931219e-02, 1.51430486e-04, -1.97976311e-01]]), 0.14159620594749628,
                            4.118296999600756),
            ('C', 'A', 4): (np.array([[8.69087540e-07, -8.24349630e-08, 1.34237570e-02],
                                      [3.36308589e-06, -1.29840012e-07, -9.87581683e-04],
                                      [-1.49294731e-02, 5.91487648e-04, 1.58688279e-01]]), -0.017219615382833137,
                            3.5280186418925132),
            ('C', 'B', 1): (np.array([[5.74275849e-07, 2.27019599e-07, -1.05353390e-02],
                                      [7.44412652e-08, -2.77061025e-10, -4.85988386e-05],
                                      [1.00569906e-02, -1.88727560e-04, 1.42547004e-01]]), 0.07867992095006568,
                            3.0928105806584294),
            ('C', 'B', 2): (np.array([[5.25184901e-06, -4.20641661e-06, -2.00013053e-02],
                                      [-7.03579769e-06, 8.81549320e-08, 2.25037229e-03],
                                      [1.99176342e-02, 1.09573796e-03, -4.17869055e-01]]), 0.5281750571312332,
                            2.1723010444670163),
            ('C', 'B', 3): (np.array([[3.29348025e-07, 3.13290114e-07, -1.10559314e-02],
                                      [4.54864082e-07, -4.23671631e-09, -5.99072501e-05],
                                      [1.05502963e-02, -1.41546423e-04, 1.17026815e-01]]), 0.21682143382062896,
                            4.055402114768905),
            ('C', 'B', 4): (np.array([[3.19197441e-06, 5.09344158e-07, -1.39997945e-02],
                                      [3.94077699e-07, -2.86996467e-09, -3.31136045e-05],
                                      [1.16246692e-02, -1.36967414e-04, 3.71838200e-01]]), -1.0880981911735053,
                            3.2436267397779925),
            ('C', 'D', 1): (np.array([[4.38128592e-07, 1.06589517e-06, -1.10977000e-02],
                                      [2.54116090e-07, -7.99504397e-09, -2.70987498e-05],
                                      [1.03196788e-02, -4.55202180e-04, 2.35472040e-01]]), 1.8810482696880706,
                            5.311175580012618),
            ('C', 'D', 2): (np.array([[9.68795261e-06, 4.66548968e-06, -2.54563830e-02],
                                      [-6.97979461e-06, 1.88206755e-07, 2.05455096e-03],
                                      [1.92919129e-02, -2.01618600e-03, 1.18515415e+00]]), 0.9664900198985676,
                            3.0573652800729514),
            ('C', 'D', 3): (np.array([[-7.13047117e-07, 1.95157152e-06, 1.10473647e-02],
                                      [1.57074906e-06, 4.41125985e-09, -5.05596705e-04],
                                      [-1.17753301e-02, -6.58801911e-04, 3.25302617e-01]]), -1.5666239035894076,
                            6.141123361481922),
            ('C', 'D', 4): (np.array([[8.59017403e-07, 3.53872135e-06, -1.44078487e-02],
                                      [3.83739283e-06, 1.47381710e-07, -1.66127522e-03],
                                      [1.12858382e-02, -1.03292933e-03, 9.85144540e-01]]), 1.0940014594845542,
                            4.21160617479416),
            ('D', 'A', 1): (np.array([[-8.02539461e-08, -1.35836905e-06, 1.09014519e-02],
                                      [2.34329317e-06, 3.97573931e-09, -8.29382265e-04],
                                      [-1.11921256e-02, 4.61893510e-04, 1.11722859e-01]]), 0.8998493555858764,
                            3.784020238353342),
            ('D', 'A', 2): (np.array([[1.15935879e-05, 1.14203216e-05, -2.94456459e-02],
                                      [-6.87737610e-06, -3.10031415e-07, 3.22827572e-03],
                                      [2.05059905e-02, -4.15439144e-03, 1.56109421e+00]]), -0.5081076788384974,
                            3.13865911335172),
            ('D', 'A', 3): (np.array([[2.24817508e-07, -6.92959917e-06, -1.01244523e-02],
                                      [1.91854199e-06, 9.88461938e-09, -5.42769473e-04],
                                      [1.14949191e-02, 2.01428581e-03, -4.13982021e-01]]), -0.6627156595475379,
                            4.504198887599453),
            ('D', 'A', 4): (np.array([[-2.14811781e-07, -2.09570850e-06, -1.04368487e-02],
                                      [3.30381113e-06, 3.99480859e-08, -1.19459427e-03],
                                      [1.02585473e-02, 8.74510327e-04, 6.06082261e-02]]), -1.0039836970281002,
                            2.730519631832975),
            ('D', 'B', 1): (np.array([[-3.81091352e-07, -1.06578095e-07, -9.86106628e-03],
                                      [-2.95171076e-07, 6.94422858e-09, 2.55585257e-04],
                                      [1.02822418e-02, -1.41987551e-04, -9.08094064e-02]]), -1.6894215372413224,
                            3.557990627555916),
            ('D', 'B', 2): (np.array([[5.93913882e-06, -1.90086641e-05, 2.36204666e-02],
                                      [2.46217259e-05, -1.40405459e-07, -8.06160889e-03],
                                      [-2.90649540e-02, 6.43500822e-03, 1.06972129e+00]]), 0.3939760344450597,
                            2.8409123805751406),
            ('D', 'B', 3): (np.array([[5.45062905e-07, -1.36342319e-07, 1.07439771e-02],
                                      [1.27555615e-06, 2.38676973e-08, -2.21960691e-04],
                                      [-1.13307389e-02, -1.87130991e-04, 1.31050900e-01]]), 0.43557786319948477,
                            4.525729597548589),
            ('D', 'B', 4): (np.array([[-1.19898465e-06, 1.62799195e-05, -1.48893878e-02],
                                      [-1.84420908e-05, 2.01989698e-07, 7.87537477e-03],
                                      [1.62367513e-02, -7.18620917e-03, -2.98094954e-01]]), -0.7088220967150424,
                            3.7208145973646887),
            ('D', 'C', 1): (np.array([[6.64150921e-08, 2.72874992e-07, -1.03066494e-02],
                                      [-4.16126389e-07, -9.87432753e-10, 1.09595503e-04],
                                      [1.03931191e-02, -4.67774611e-05, -4.19556131e-02]]), -1.9184124755111516,
                            5.219067831447366),
            ('D', 'C', 2): (np.array([[3.72171867e-06, 1.26576084e-05, -2.50727901e-02],
                                      [-3.47590415e-06, -3.86459518e-07, 1.91361135e-03],
                                      [2.01930255e-02, -4.18886605e-03, 9.81755972e-01]]), -0.36202404725895915,
                            3.3659456137724417),
            ('D', 'C', 3): (np.array([[5.99663737e-07, -2.93921394e-06, 1.31779892e-02],
                                      [-6.96507916e-06, 2.84278669e-08, 2.04466636e-03],
                                      [-1.02763011e-02, 9.08195055e-04, -9.22304508e-01]]), 1.1294191584727113,
                            5.960453878007358),
            ('D', 'C', 4): (np.array([[8.33792353e-06, 5.80033116e-06, 8.69666269e-03],
                                      [1.39940590e-05, -1.03665120e-08, -5.32594000e-03],
                                      [-2.16891569e-02, -2.20100976e-03, 3.72232429e+00]]), 1.254158912335075,
                            6.58554087255596)}


if __name__ == '__main__':
    gt_path = "/home/brian/Documents/datasets/new_smith_full/db4_test.json"
    p = get_parser_from_arguments()
    preds = test_multi_view(p, ("A", "B", "C", "D"), f_matrices=get_matrices())
    # preds = test_single_view2(p, ("A", "B", "C", "D"))
    stats = evaluation.evaluate(gt_path, preds)
    # print("Weak th = 0.001, score th = 0.3")
    # preds = test_multi_view(p, ("A", "B", "C", "D"), weak_th=0.001, f_matrices=get_matrices())
    # stats = evaluation.evaluate(gt_path, preds)
    # print("Weak th = 0.1, score th = 0.3")
    # preds = test_multi_view(p, ("A", "B", "C", "D"), weak_th=0.1, f_matrices=get_matrices())
    # stats = evaluation.evaluate(gt_path, preds)
    # print("Weak th = 0.01, score th = 0.2")
    # preds = test_multi_view(p, ("A", "B", "C", "D"), weak_th=0.01, score_th=0.2, f_matrices=get_matrices())
    # stats = evaluation.evaluate(gt_path, preds)
    # print("Weak th = 0.01, score th = 0.1")
    # preds = test_multi_view(p, ("A", "B", "C", "D"), weak_th=0.01, score_th=0.1, f_matrices=get_matrices())
    # stats = evaluation.evaluate(gt_path, preds)
    # print("Weak th = 0.01, score th = 0.4")
    # preds = test_multi_view(p, ("A", "B", "C", "D"), weak_th=0.01, score_th=0.4, f_matrices=get_matrices())
    # stats = evaluation.evaluate(gt_path, preds)
    # print("Weak th = 0.01, score th = 0.5")
    # preds = test_multi_view(p, ("A", "B", "C", "D"), weak_th=0.01, score_th=0.5, f_matrices=get_matrices())
    # stats = evaluation.evaluate(gt_path, preds)
    # json_file_name = os.path.join(parser.db_name, parser.test["json_file_output"])
    #
    # with open(json_file_name, 'w') as f:
    #     json.dump(detections, f)
