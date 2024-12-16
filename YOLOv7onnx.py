#%%
# !pip install --upgrade setuptools pip --user
# !pip install onnx
# !pip install onnxruntime
# #!pip install --ignore-installed PyYAML
# #!pip install Pillow
#
# !pip install protobuf<4.21.3
# !pip install onnxruntime-gpu
# !pip install onnx>=1.9.0
# !pip install onnx-simplifier>=0.3.6 --user
#%%
import sys
import torch
from utils.torch_utils import select_device, time_synchronized, TracedModel
from tqdm import tqdm
from torchvision.ops import batched_nms
import matplotlib.pyplot as plt

from utils.general import xywh2xyxy
from collections import defaultdict
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

# Compute precision, recall, and AP for a single class
def compute_precision_recall_ap(preds, gts, iou_threshold=0.5):
    preds = sorted(preds, key=lambda x: x[2], reverse=True)  # Sort by confidence

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    matched = set()

    for i, (pred_box, pred_cls, confidence) in enumerate(preds):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, (gt_box, gt_cls) in enumerate(gts):
            if gt_idx in matched:
                continue

            if pred_cls == gt_cls:
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            tp[i] = 1
            matched.add(best_gt_idx)
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / len(gts)

    # Compute AP using the precision-recall curve
    ap = 0
    for i in range(1, len(precision)):
        ap += (recall[i] - recall[i - 1]) * precision[i]

    return precision, recall, ap

# Compute mAP and plot precision-recall curve
def compute_map(predictions, ground_truths, num_classes, iou_threshold=0.5):
    aps = []
    plt.figure(figsize=(10, 7))

    for cls in range(num_classes):
        preds = [(box, cls_id, conf) for box, cls_id, conf in predictions if cls_id == cls]
        gts = [(box, cls_id) for box, cls_id in ground_truths if cls_id == cls]

        if len(gts) == 0:
            continue

        precision, recall, ap = compute_precision_recall_ap(preds, gts, iou_threshold)
        aps.append(ap)

        plt.plot(recall, precision, label=f"Class {cls} (AP={ap:.2f})")

    mAP = np.mean(aps)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (mAP={mAP:.2f})")
    plt.legend()
    plt.grid()
    plt.show()

    return mAP

def yolobbox_to_xyxy(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    x2 = x1 + w
    y2 = y1 + h
    return [int(x1), int(y1), int(x2), int(y2)]

coco_lass_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
#%%
# !nvidia-smi
# #%%
# !# Download YOLOv7 code
# !git clone https://github.com/WongKinYiu/yolov7
# %cd yolov7
# !ls
# #%%
# !# Download trained weights
# !wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
# #%%
# !python detect.py --weights ./yolov7-tiny.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
#%%
from PIL import Image
# Image.open('/content/yolov7/runs/detect/exp/horses.jpg')
#%%
# export ONNX for ONNX inference
# %cd /content/yolov7/
# !python export.py --weights ./yolov7-tiny.pt \
#         --grid --end2end --simplify \
#         --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 \
#         --img-size 640 640 --max-wh 640 # For onnxruntime, you need to specify this value as an integer, when it is 0 it means agnostic NMS,
#                      # otherwise it is non-agnostic NMS
#%%
# show ONNX model
# !ls
#%%
# Inference for ONNX model
import cv2
#%%
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
import argparse

from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr

test_vectors_test = False
reshape = False
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def main(opt):

    # Model
    cuda = [True if torch.cuda.is_available() else False][0]
    w = '/mnt/Data/hanoch/tir_old_tf/tir_od_1.5.onnx'

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(w, providers=providers)
    input_shape = session.get_inputs()[0].shape

    # sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.intra_op_num_threads = 8  # Limit the number of threads
    # sess = ort.InferenceSession(onnx_file, sess_options, providers=["CPUExecutionProvider"])


    test_path = '/home/hanoch/projects/tir_od/yolov7/tir_od/test_set/Test51a_Test40A_test_set.txt'
    imgsz_test = 512 #(512, 512, 3)
    dataset_image_size = [640, 640]
    batch_size = 16
    gs = 32 #gs = max(int(model.stride.max()), 32)

    uniq_class_subset = ['Car', 'Person']
    coco_mapping = {ix : coco_lass_names.index(x.lower()) for ix, x in  enumerate(uniq_class_subset)}

    opt.device = torch.device("cuda:" + str(opt.device) + "" if torch.cuda.is_available() else "cpu")

    hyp = dict()
    hyp['person_size_small_medium_th'] = 32 * 32
    hyp['car_size_small_medium_th'] = 44 * 44

    hyp['img_percentile_removal'] = 0.3
    hyp['beta'] = 0.3
    hyp['gamma'] = 80  # dummy anyway augmentation is disabled
    hyp['gamma_liklihood'] = 0.01
    hyp['random_pad'] = True
    hyp['copy_paste'] = False
    nc = 2

    images_parent_folder = '/mnt/Data/hanoch/tir_frames_rois/yolo7_tir_data_all'

    print(colorstr('val: '))
    opt.single_cls = False
    opt.norm_type = 'no_norm'
    opt.input_channels = 1
    opt.tir_channel_expansion = False
    opt.no_tir_signal = False
    dataloader = create_dataloader(test_path, imgsz_test, batch_size, gs, opt,  # testloader
                                                 hyp=hyp, cache=opt.cache_images, rect=False, rank=-1,
                                                 # @@@ rect was True why?
                                                 world_size=1, workers=8,
                                                 pad=0.5, prefix=colorstr('val: '),
                                                 rel_path_images=images_parent_folder, num_cls=nc)[0]

    seen = 0
    t0 = 0

    test_vector_image_resolution = [768, 1024]
    test_vector_ref_detections = { 'BBox': [383.031, 248.378, 39.7324, 87.0982], 'Score': 0.982154, 'Class': 'Person', 'crop_coordination_xy':[256, 128], 'tol_box':0.1, 'tol_conf':0.2}
    test_image = '/mnt/Data/hanoch/tir_old_tf/ODFrame_prod_onnx_tir_v1_tf_test_vector.png'

    det_threshold = 0.8
    nms_ious_th = 0.5

    predictions = list()
    ground_truths = list()

    for img, targets, paths, shapes in tqdm(dataloader):
        img = img.cpu().numpy().astype('float32').transpose(0, 2,3,1) # channel last
        nb, _, height, width = img.shape  # batch size, channels, height, width
        seen += 1
        with (torch.no_grad()):
            # Run model
            t = time_synchronized()
            outname = [i.name for i in session.get_outputs()]
            inname = [i.name for i in session.get_inputs()]
            # print(inname, outname)

            for ix, im in enumerate(img):
                if test_vectors_test:
                    im = cv2.imread(test_image)  # BGR
                    if not(opt.no_tir_signal):
                        im = im[:, :, :1]  # channels are duplicated in the source
                        # im = im[:, :, np.newaxis]
                        if reshape:
                            im = cv2.resize(im, (imgsz_test, imgsz_test))[:, :, np.newaxis]
                        else:
                            # im = im[test_vector_ref_detections['crop_coordination'][0]:test_vector_ref_detections['crop_coordination'][0]+512,
                            #      test_vector_ref_detections['crop_coordination'][1]:test_vector_ref_detections['crop_coordination'][1]+512]
                            # im = im*255
                            im = im[test_vector_ref_detections['crop_coordination_xy'][1]:
                                    test_vector_ref_detections['crop_coordination_xy'][1] + 512,
                                     test_vector_ref_detections['crop_coordination_xy'][0]:
                                     test_vector_ref_detections['crop_coordination_xy'][0] + 512]
                        if 0:
                            im = im/255.0
                        im = im.astype('float32')
                else:
                    im = im /(2**16 -1)
                    im = im.astype('float32')

                inp = {inname[0]: im[np.newaxis,:,:,:]}
                output = session.run(outname, inp)
                sigmoid_out = output[0]
                # BBOX CV2 is upper x,y and w, h while YOLO is center x,y
                bbox_out = output[1]
                class_id = np.argmax(sigmoid_out, 2)
                scores = np.max(sigmoid_out, 2)
                scores_above_th = scores[scores > det_threshold]
                bboxes = bbox_out[scores>det_threshold]
                ml_class_id = class_id[scores > det_threshold].astype('int')

                if test_vectors_test: # since no labels file annotations are not resized like dat is implicitly
                    if reshape:
                        bboxes_normalized_coord_detected = [[int(i[0].item() * test_vector_image_resolution[1] / imgsz_test),
                                              int(i[1].item() * test_vector_image_resolution[0] / imgsz_test),
                                              int(i[2].item() * test_vector_image_resolution[1] / imgsz_test),
                                              int(i[3].item() * test_vector_image_resolution[0] / imgsz_test)] for i in bboxes]
                    else:
                        bboxes_normalized_coord_detected = bboxes
                        bboxes_normalized_coord_detected[:, 0] += test_vector_ref_detections['crop_coordination_xy'][0]
                        bboxes_normalized_coord_detected[:, 1] += test_vector_ref_detections['crop_coordination_xy'][1]

                    norm_err = [(b - np.array(test_vector_ref_detections['BBox'])) / np.array(test_vector_ref_detections['BBox'])
                                for b in bboxes_normalized_coord_detected]

                    good_enough = [np.isclose(b, np.array(test_vector_ref_detections['BBox']), rtol=0.02).all() # 2% relative tolerance
                                   # 2% relative tolerance
                                   for b in bboxes_normalized_coord_detected]

                    good_enough_abs_tol = [np.isclose(b, np.array(test_vector_ref_detections['BBox']), atol=test_vector_ref_detections['tol_box']).all() # 2% relative tolerance
                                   # 2% relative tolerance
                                   for b in bboxes_normalized_coord_detected]

                    if np.array([not (i) for i in good_enough]).any():
                        print('!!!!!!!!!!!!! Heuston we have a problem BBOX relative tolerance')

                    if not((np.isclose(scores_above_th, test_vector_ref_detections['Score'])).any()):
                        print('!!!!!!!!!!!!! Heuston we have a problem conf relative tolerance')

                else:
                    # bboxes_normalized_coord_detected = [[int(i[0].item() * dataset_image_size[1] / imgsz_test), # GT labels are related to 640*640 while NN output isrelated to 512
                    #                       int(i[1].item() * dataset_image_size[0] / imgsz_test),
                    #                       int(i[2].item() * dataset_image_size[1] / imgsz_test),
                    #                       int(i[3].item() * dataset_image_size[0] / imgsz_test)] for i in bboxes]
                    # bboxes_normalized_coord_detected = bboxes
                    # cv2: xywh to xyxyx
                    bboxes_normalized_coord_detected_2_nms = [[int(i[0].item()), # GT labels are related to 640*640 while NN output isrelated to 512
                                           int(i[1].item()), int(i[0].item() + i[2].item()),
                                           int(i[1].item() + i[3].item())] for i in bboxes]
                    # b = xywh2xyxy(bboxes_normalized_coord_detected.reshape(-1, 4)).ravel().astype(np.int)


                    gt = [[0.852344, 0.082031, 0.126562, 0.107813], # 0
                          [0.685937, 0.079687, 0.021875, 0.081250], # 1
                          [0.922656, 0.067187, 0.051562, 0.050000]] # 0

                    # [yolobbox_to_xyxy(gt[0], gt[1], gt[2], gt[3], 640, 640) for gt in gt]
                    # [[505, 17, 586, 87] = > CaR , [431, 24, 445, 76], [574, 26, 606, 58]]

                    bboxes_normalized_coord_detected_2_nms = torch.tensor(bboxes_normalized_coord_detected_2_nms)
                    bboxes_normalized_coord_detected_2_nms = bboxes_normalized_coord_detected_2_nms.to(torch.float)

                #     list of detections, on (n,6) tensor per image [xyxy, conf, cls]

                nms_out = batched_nms(boxes=bboxes_normalized_coord_detected_2_nms,
                                      scores=torch.tensor(scores_above_th),
                                      idxs=torch.tensor(ml_class_id),
                                      iou_threshold=nms_ious_th)
                preds = [(bboxes_normalized_coord_detected_2_nms[x].numpy(), ml_class_id[x].item(), scores_above_th[x].item()) for x in nms_out]
                predictions.extend(preds)

                img_gt_lbls = targets[targets[:, 0] == ix, 1].cpu().numpy()
                img_gt_lbls_in_coco_order = [coco_mapping[ele.item()] for ele in img_gt_lbls]
                img_gt_boxes = targets[targets[:, 0] == ix, 2:] # YOLO bbox are relative coordination (center x,y and w,h ) hence resolution independant
                img_gt_boxes_xyxy = [yolobbox_to_xyxy(*yolo_bb, imgsz_test, imgsz_test) for yolo_bb in img_gt_boxes]
                # img_gt_boxes_xywh = [[x[0], x[1],x[2]-x[0], x[3]-x[1]] for x in img_gt_boxes_xyxy]

                # Ground truths: [(bbox, class_id)]
                ground_truths.extend([(x,y) for x, y in zip (img_gt_boxes_xyxy, img_gt_lbls_in_coco_order)])
            t0 += time_synchronized() - t
            # out coco 80 classes : [1, 25200, 85] [batch, proposals_3_scales,4_box__coord+1_obj_score + n x classes]
            # Compute loss
            # if compute_loss:
            #     loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

        # t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz_test, imgsz_test, batch_size)  # tuple
        # if 1:
        #     print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)
        # Predicted: [(bbox, class_id, confidence)]
    map = compute_map(predictions, ground_truths, num_classes=3, iou_threshold=0.5)
    print(f"mAP: {map:.2f}")

    return

if __name__ == '__main__':

    print(f"Python version: {sys.version}, {sys.version_info} ")
    print(f"Pytorch version: {torch.__version__} ")

    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    main(opt=opt)