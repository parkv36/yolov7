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

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
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

    cuda = [True if torch.cuda.is_available() else False][0]
    w = '/mnt/Data/hanoch/tir_old_tf/tir_od_1.5.onnx'
    img = cv2.imread('/home/hanoch/projects/tir_od/yolov7/inference/images/horses.jpg')

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(w, providers=providers)

    # sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.intra_op_num_threads = 8  # Limit the number of threads
    # sess = ort.InferenceSession(onnx_file, sess_options, providers=["CPUExecutionProvider"])


    test_path = '/home/hanoch/projects/tir_od/yolov7/tir_od/test_set/Test51a_Test40A_test_set.txt'
    imgsz_test = 512 #(512, 512, 3)
    batch_size = 16
    gs = 32 #gs = max(int(model.stride.max()), 32)
    input_shape = session.get_inputs()[0].shape

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
    dataloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
                                                 hyp=hyp, cache=opt.cache_images, rect=False, rank=-1,
                                                 # @@@ rect was True why?
                                                 world_size=1, workers=8,
                                                 pad=0.5, prefix=colorstr('val: '),
                                                 rel_path_images=images_parent_folder, num_cls=nc)[0]

    seen = 0
    t0 = 0

    test_vector = { 'BBox': [383.031, 248.378, 39.7324, 87.0982], 'Score': 0.982154, 'Class': 'Person'}
    test_image = '/mnt/Data/hanoch/tir_old_tf'
    det_threshold = 0.8
    nms_th = 0.5
    for img, targets, paths, shapes in tqdm(dataloader):
        img = img.cpu().numpy().astype('float32').transpose(0, 2,3,1) # channel last
        # img = img.to(opt.device, non_blocking=True)
        # img = img.half() if half else img.float()
        # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0 c# already done inside dataloader
        targets = targets.to(opt.device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        seen += 1

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            outname = [i.name for i in session.get_outputs()]
            inname = [i.name for i in session.get_inputs()]
            print(inname, outname)

            for im in img:
                inp = {inname[0]: im[np.newaxis,:,:,:]}
                output = session.run(outname, inp)
                sigmoid_out = output[0]
                bbox_out = output[1]
                class_id = np.argmax(sigmoid_out, 2)
                scores = np.max(sigmoid_out, 2)
                bboxes = bbox_out[scores>det_threshold]

            # %% md
            # # ONNX inference

            # out, train_out = model(img, augment=False)  # inference out [batch, proposals, figures_of] figures_of :(4 coordination, obj conf, cls conf ) and training outputs(batch_size, anchor per scale, x,y dim of scale out 40x40 ,n_classes-conf+1-objectness+4-bbox ) over 3 scales diferent outputs (2,2,80,80,7), (2,2,40,40,7)  : 640/8=40
            t0 += time_synchronized() - t
            # out coco 80 classes : [1, 25200, 85] [batch, proposals_3_scales,4_box__coord+1_obj_score + n x classes]
            # Compute loss
            # if compute_loss:
            #     loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz_test, imgsz_test, batch_size)  # tuple
        if 1:
            print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)


    colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    im.shape


    outname = [i.name for i in session.get_outputs()]
    outname

    inname = [i.name for i in session.get_inputs()]
    inname

    inp = {inname[0]:im}
    #%% md
    # # ONNX inference
    outputs = session.run(outname, inp)[0]
    # outputs
    #%%
    ori_images = [img.copy()]

    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score),3)
        name = names[cls_id]
        color = colors[name]
        name += ' '+str(score)
        cv2.rectangle(image,box[:2],box[2:],color,2)
        cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)

    Image.fromarray(ori_images[0])

    return

if __name__ == '__main__':

    print(f"Python version: {sys.version}, {sys.version_info} ")
    print(f"Pytorch version: {torch.__version__} ")

    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    main(opt=opt)