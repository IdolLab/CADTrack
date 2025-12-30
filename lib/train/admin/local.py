class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/sqh/lihao/CADTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/sqh/lihao/CADTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/sqh/lihao/CADTrack/pretrained_networks'
        self.got10k_val_dir = '/home/sqh/lihao/CADTrack/data/got10k/val'
        self.lasot_lmdb_dir = '/home/sqh/lihao/CADTrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/sqh/lihao/CADTrack/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/home/sqh/lihao/CADTrack/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/home/sqh/lihao/CADTrack/data/coco_lmdb'
        self.coco_dir = '/home/sqh/lihao/CADTrack/data/coco'
        self.lasot_dir = '/home/sqh/lihao/CADTrack/data/lasot'
        self.got10k_dir = '/home/sqh/lihao/CADTrack/data/got10k/train'
        self.trackingnet_dir = '/home/sqh/lihao/CADTrack/data/trackingnet'
        self.depthtrack_dir = '/home/sqh/lihao/CADTrack/data/depthtrack/train'
        self.lasher_dir = '/home/sqh/lihao/CADTrack/data/RGBTdatasets/LasHeR/train'
        self.vtuav_dir = '/home/sqh/lihao/STTrack/data/RGBTdatasets/VTUAV'
        self.visevent_dir = '/home/sqh/lihao/CADTrack/data/visevent/train'
