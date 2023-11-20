class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/richard_lei/workspace'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/home/richard_lei/dataset/LASOT/LaSOTBenchmark'
        self.got10k_dir = '/home/richard_lei/dataset/GOT/train_data'
        self.trackingnet_dir = '/home/richard_lei/dataset/TrackingNet'
        self.coco_dir = '/home/richard_lei/dataset/COCO'
        self.lvis_dir = ''
        self.sbd_dir = ''

        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''

