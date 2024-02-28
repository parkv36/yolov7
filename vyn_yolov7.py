from .train import run_train


class Setting:
    def __init__(self):
        self.weights = 'yolo7.pt'
        self.cfg = ''
        self.data = 'data/coco.yaml'
        self.hyp = 'data/hyp.scratch.p5.yaml'
        self.epochs = 300
        self.batch_size = 16
        self.img_size = [640, 640]
        self.rect = False
        self.resume = False
        self.nosave = False
        self.notest = False
        self.noautoanchor = False
        self.evolve = False
        self.bucket = ''
        self.cache_images = False
        self.image_weights = False
        self.device = ''
        self.multi_scale = False
        self.single_cls = False
        self.adam = False
        self.sync_bn = False
        self.local_rank = -1
        self.workers = 8
        self.project = 'runs/train'
        self.entity = None
        self.name = 'exp'
        self.exist_ok = False
        self.quad = False
        self.linear_lr = False
        self.label_smoothing = 0.0
        self.upload_dataset = False
        self.bbox_interval = -1
        self.save_period = -1
        self.artifact_alias = 'latest'
        self.freeze = [0]
        self.v5_metric = False
        self.normalise_image = True
        self.redo_caching = False
        self.custom_augment_fun = None

        # TEST
        self.conf_thres = 0.001
        self.iou_thres = 0.6

        self.model = None
        self.preprocessing_function = None
        self.generators = None


def set_and_run_train(options=None, **kwargs):
    if options is None:
        options = Setting()

        for key in kwargs.keys():
            if hasattr(options, key):
                setattr(options, key, kwargs[key])

    run_train(options)


if __name__ == '__main__':
    kwargs = {'project': '/home/isaac/Downloads/ERASE',
              'name': 'model_erase',
              'device': '0',
              'data': '/media/isaac/Data/ubuntu/containers/yolov7/data/vyn_coco.yaml',
              'hyp': '/media/isaac/Data/ubuntu/containers/yolov7/data/hyp.scratch.p5.yaml',
              'cfg': '/media/isaac/Data/ubuntu/containers/yolov7/cfg/training/yolov7.yaml',
              'batch_size': 8,
              'weights': ''}

    set_and_run_train(options=None, **kwargs)
