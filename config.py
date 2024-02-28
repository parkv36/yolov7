import json
import os
from typing import List, Union

import numpy as np

from ai_model_utils.base_configuration import (BaseConfiguration,
                                               BaseConfigurationParameters,
                                               BaseModelParameters,
                                               BaseStandardParameters,
                                               BaseTrainingParameters)
from ai_data_processing.types.object_detection_model_name import ObjectDetectionModelName
from ai_model_utils.types.framework_and_models import FrameworkType


class Yolov7TrainingParameters(BaseTrainingParameters):
    """
    base parameters for training models
    """

    def __init__(self, num_classes=-1) -> None:
        super().__init__(num_classes)

        self.batch_size = 3
        self.epochs = 60
        self.first_stage_epoch = 26
        self.iterations = 800 // self.batch_size
        self.min_box_size = 15
        self.max_bbox_per_scale = 150
        self.only_backbone = False
        #self.positive_conf_weight = 0.5
        #self.score_thresh_train = 0.4  # Not used currently, this will be useful for training
        self.score_thresh = 0.4  # Needs checking
        self.smooth_labels_weight = 0.01
        #self.until_neck = True
        self.warmup_steps = 3

        self.max_stride = [32, 32]
        # Whether to normalise the image within the yolov7 framework. This means that the image is not normalised
        # inside the preprocessing function. This is common for the yolov7 framework but not for our generator
        # So, normally this is True for yolov7 pipeline and False when using our generator. The option None follows this
        # pattern and set the value as self.normalise_image_yolov7 = not self.use_vyn_generator in the runner
        self.normalise_image_yolov7 = None

        # Domain adaptation
        self.domain_threshold_model = 1
        self.domain_threshold_discriminator = 2

        self.losses = {}
        self.metrics = {}

        # These are names of a class object that should not be used for training. For instance,
        # if a model to detect barriers has very blurry or small barriers, the labelling or not
        # labelling will harm the model, adding them in this list will make the model not to use them
        self.remove_names = []
        self.hyperparameter_filename = None
        self.initial_weights = None  # '/media/isaac/Data/ubuntu/containers/Data/humanlearning/Detection/Model/pretrained'

        self.use_vyn_generator = False

        # use_image_weights does not seem to be used. In general, this is a poorer version of our class rate
        # So do not use unless we want to check something
        self.use_image_weights = False
        # do_hyperparameter_evolution: It performs a genetic optimisation to find the best augmentation parameters
        self.do_hyperparameter_evolution = False
        # Different image scale
        self.do_image_multi_scale = False
        # Rectangular training gets the ration h/w of the images and order them according to this value, so that
        # batches are selected from low to high. It also changes the size of the image to the closest that is divisible
        # by the stride value.
        self.use_rectangular_training = False
        # Layers to freeze since the beginning
        self.freeze_layers = [0]
        self.num_workers = 8

        self.iou_loss_threshold = 0.2  # IoU training threshold
        self.iou_thresh_test = 0.25  # Needs checking
        self.train_lr_init = 0.01    # initial learning rate (SGD=1E-2, Adam=1E-3)
        self.train_lr_end = 0.2  # final OneCycleLR learning rate (lr0 * lrf)
        self.momentum = 0.937  # SGD momentum/Adam beta1

        self.weight_decay = 0.0005  # optimizer weight decay 5e-4
        self.warmup_epochs = 3.0  # warmup epochs (fractions ok)
        self.warmup_momentum = 0.8  # warmup initial momentum
        self.warmup_bias_lr = 0.1  # warmup initial bias lr
        # The next three parameters are the weights for the 3 parts of the loss: bb, confidence (obj) and classes
        self.box = 0.05  # box loss gain
        self.cls = 0.3  # cls loss gain
        self.obj = 0.7  # obj loss gain (scale with pixels)
        # The next two parameters are used for the BCE loss as the positive weight, for instance if we have a dataset
        # of 100 positive cases and 300 negative, this value should 3 to make positives to be more important.
        self.cls_pw = 1.0  # cls BCELoss positive_weight
        self.obj_pw = 1.0  # obj BCELoss positive_weight

        # 1/self.anchor_t is the threshold for the Kmeans clustering for anchor boxes to be considered correct. So, when
        # the fitting the current method uses the division between the ratios (cluster and real data) and if this
        # division is smaller than 1 / self.anchor_t is not considered as a good fit.
        self.anchor_t = 4.0  # anchor-multiple threshold
        # self.anchors = 3  # anchors per output layer (0 to ignore)
        self.fl_gamma = 0.0  # focal loss gamma (efficientDet default gamma=1.5)
        self.fl_alpha = 0.25  # focal loss alpha (efficientDet default alpha=0.25)

        # AUGMENTATION
        self.hsv_h = 0.015  # image HSV-Hue augmentation (fraction)
        self.hsv_s = 0.7  # image HSV-Saturation augmentation (fraction)
        self.hsv_v = 0.4  # image HSV-Value augmentation (fraction)
        self.degrees = 0.0  # image rotation (+/- deg)
        self.translate = 0.2  # image translation (+/- fraction)
        self.scale = 0.9  # image scale (+/- gain)
        self.shear = 0.0  # image shear (+/- deg)
        self.perspective = 0.0  # image perspective (+/- fraction), range 0-0.001
        self.flipud = 0.0  # image flip up-down (probability)
        self.fliplr = 0.5  # image flip left-right (probability)
        # self.mosaic = 1.0  # image mosaic (probability)
        self.mixup = 0.15  # image mixup (probability)
        self.copy_paste = 0.0  # image copy paste (probability)
        self.paste_in = 0.15  # image copy paste (probability), use 0 for faster training
        self.loss_ota = 1  # use ComputeLossOTA, use 0 for faster training


class Yolov7StandardParameters(BaseStandardParameters):
    """
    Base standard parameters like the name of the labels
    """

    def __init__(self, data_yaml, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.labels_to_keep = {}
        self.data_yaml = data_yaml
        self.rename_class_names = {}
        self.is_class_agnostic_nms = True
        self.load_best_weights = True
        self.resume = False


class Yolov7Parameters(BaseModelParameters):
    """
    Base model parameters for object detector
    """

    def __init__(self,
                 submodel='',
                 anchors_file='',
                 anchor_base: List[float] = None,
                 anchors: List[float] = None,
                 image_shape: List[int] = None,
                 keep_strides=False,
                 ratios: List[List[Union[float, int]]] = None,
                 sizes: List[int] = None) -> None:
        super().__init__(model='yolo', version=7, tiny=False)

        self.framework = FrameworkType.PYTORCH.value
        self.submodel = submodel
        self.model_name = ObjectDetectionModelName.YOLOv7.value
        root = os.path.dirname(__file__)
        self.path_to_cfg_files = {'training': os.path.join(root, 'cfg', 'training'),
                                  'deploy': os.path.join(root, 'cfg', 'deploy')}
        self.original_cfg_filename_training = os.path.join(self.path_to_cfg_files['training'],
                                                           f'yolov7{self.submodel}.yaml')
        self.original_cfg_filename_deploy = os.path.join(self.path_to_cfg_files['deploy'],
                                                           f'yolov7{self.submodel}.yaml')

        self.strides = None
        self.xy_scales = None
        self.base_anchors = []

        self.use_spp = False
        self.noise_std = 0.
        self.use_nms = True

        self._max_bb_sizes_per_scale = 3
        self._anchors_file = anchors_file
        self.anchor_base = anchor_base
        self._anchors = anchors
        self.image_shape = image_shape
        self.anchors_per_scale = None
        # self._max_bb_sizes_per_scale = [0.15, 0.4, 1.0]  # Maximum size of bb for anchor less approach

        self._cfg_filename_training = None  # This needs to be specified at some point (the runner does it for you though)
        self._cfg_filename_deploy = None
        # self.ms_dayolo = None
        # self.add_dayolo_dan_to_warmup = False

        self.update_values(keep_strides, ratios, sizes)

    def _get_cfg_proper_return(self, cfg_filename):
        if cfg_filename is None or self.dirname is None:
            return cfg_filename
        cfg_dirname = os.path.basename(self.dirname)
        return os.path.join(self.dirname, cfg_filename[cfg_filename.find(cfg_dirname) + len(cfg_dirname)+1:])

    @property
    def cfg_filename_training(self):
        return self._get_cfg_proper_return(self._cfg_filename_training)

    @cfg_filename_training.setter
    def cfg_filename_training(self, cfg_filename):
        self._cfg_filename_training = cfg_filename
        self.has_changed = True

    @property
    def cfg_filename_deploy(self):
        return self._get_cfg_proper_return(self._cfg_filename_deploy)

    @cfg_filename_deploy.setter
    def cfg_filename_deploy(self, cfg_filename):
        self._cfg_filename_deploy = cfg_filename
        self.has_changed = True

    @property
    def anchors(self):
        return self._anchors

    @anchors.setter
    def anchors(self, anchors):
        self._anchors = anchors
        if anchors is None:
            self.anchors_per_scale = None

    @property
    def anchors_file(self):
        return self._anchors_file

    @anchors_file.setter
    def anchors_file(self, anchors_file):
        self._anchors_file = anchors_file
        self.get_anchors()

    @property
    def max_bb_sizes_per_scale(self):
        return self._max_bb_sizes_per_scale

    @max_bb_sizes_per_scale.setter
    def max_bb_sizes_per_scale(self, max_bb_sizes_per_scale):
        self._max_bb_sizes_per_scale = max_bb_sizes_per_scale
        if np.max(max_bb_sizes_per_scale) <= 1:
            self.max_bb_sizes_per_scale = np.array(
                max_bb_sizes_per_scale) * max(self.image_shape)
        self.has_changed = True

    def update_values(self,
                      keep_strides=False,
                      ratios: List[List[Union[float, int]]] = None,
                      sizes: List[int] = None) -> None:
        """
        Update the values of the strides, xy_scales and anchors. This method should be used any time a variable
        is changed
        :param keep_strides: (bool) Normally, this should be false, but if the strides must be customised, then this
                                    variable can be set to 0. Notice that models may break and pre-trained models are
                                    unlikely to work. So, use it only if you know what you are doing.
        :param ratios: This is a set of rations [w, h] to create anchors, It is a list of lists
        :param sizes: This is a list with sizes each of them will multiply all the ratios, so the number of anchors is
                        len(ratios) * len(sizes)
        :return: None
        """
        if self.model == 'yolo' and not self.tiny and not keep_strides:
            self.strides = [8, 16, 32]
            if self.xy_scales is None or len(self.xy_scales) != 3:
                self.xy_scales = [1.2, 1.1, 1.05]
        elif self.model == 'yolo' and self.tiny and not keep_strides:
            self.strides = [16, 32]
            if self.xy_scales is None or len(self.xy_scales) != 2:
                self.xy_scales = [1.1, 1.05]
        elif self.backbone == 'vgg16' and not keep_strides:
            self.strides = [32]
        elif 'resnet' in self.backbone.lower() and not keep_strides:
            self.strides = [32]

        self.get_anchors(ratios, sizes)

    def get_anchors(self,
                    ratios: List[List[Union[float, int]]] = None,
                    sizes: List[int] = None) -> None:
        """
        Get the anchors of the model. Notice that this method requires the anchor_file to exist. Since there are object
        detectors that do not need anchors, anchor_file is not mandatory.
        :return:
        """
        if (self.anchors is None or len(self.anchors) == 0) and self.anchors_file and os.path.isfile(self.anchors_file):
            with open(self.anchors_file, 'r') as f:
                self.base_anchors = json.load(f)

            anchors = [
                int(np.round(self.image_shape[1 - j] * anchor_i))
                for anchor in self.base_anchors
                for j, anchor_i in enumerate(anchor)
            ]

            anchors = np.array(anchors).reshape([len(self.strides), -1, 2])
            self.anchors_per_scale = anchors.shape[1]
            self.anchors = anchors
        elif ratios is not None or sizes is not None:
            if ratios is None:
                ratios = [1, 1]
            ratios = np.array(ratios).reshape(-1, 2, 1)
            if sizes is None:
                sizes = [1]
            sizes = np.array(sizes).reshape(1, 1, -1)

            self.anchors = (ratios * sizes).transpose(0, 2,
                                                      1).reshape(1, -1, 2)
            self.anchors_per_scale = self.anchors.shape[1]

        if self._max_bb_sizes_per_scale is not None and self.image_shape is not None:
            self.max_bb_sizes_per_scale = self._max_bb_sizes_per_scale


class Yolov7Configuration(BaseConfiguration):
    """
    Base configuration for object detectors. It contains the training, standard and model parameters
    """

    def __init__(self,
                 training_parameters: Yolov7TrainingParameters = None,
                 standard_parameters: Yolov7StandardParameters = None,
                 model_parameters: Yolov7Parameters = None) -> None:

        super().__init__()
        self.name = 'yolo_config'
        self.training_parameters = training_parameters or Yolov7TrainingParameters(
        )
        self.standard_parameters = standard_parameters or Yolov7StandardParameters(
        )
        self.model_parameters = model_parameters or Yolov7Parameters()

        self.normalise_num_classes()

        # if self.model_parameters.tfod_parameters is not None:
        if FrameworkType(self.model_parameters.framework) == FrameworkType.TFOD and \
                self.model_parameters.tfod_parameters is not None:
            self.transform_tfod_pipeline_config()

    def normalise_num_classes(self):
        if self.standard_parameters.num_classes is not None and self.training_parameters.num_classes is None:
            self.training_parameters.num_classes = self.standard_parameters.num_classes
        if self.standard_parameters.num_classes is None and self.training_parameters.num_classes is not None:
            self.standard_parameters.num_classes = self.training_parameters.num_classes
        if self.standard_parameters.num_classes != self.training_parameters.num_classes:
            raise ValueError(
                'standard_parameters.num_classes must be the same as training_parameters.num_classes'
            )


def config(anchors_file: str = None,
           class_file='') -> Yolov7Configuration:
    """
    Create a standard configuration for an object detector model. This function should be created per model, since
    many of the values are going to be changed.
    :param anchors_file: The address where the anchor file is stored, if needed. Use '' when not needed
    :param model: The name of the model to use
    :param version: The version of the model to use
    :param tiny: Whether the model is tiny or not when required.
    :param class_file: The path to the file storing the class names
    :return: An ObjectDetectorConfiguration object
    """
    image_shape = [512, 512, 3]
    model_params = Yolov7Parameters(anchors_file=anchors_file,
                                    image_shape=image_shape)
    standard_params = Yolov7StandardParameters(class_file=class_file,
                                               image_shape=image_shape)
    training_params = Yolov7TrainingParameters(
        standard_params.num_classes)

    return Yolov7Configuration(training_parameters=training_params,
                                       standard_parameters=standard_params,
                                       model_parameters=model_params)
