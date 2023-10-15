import os.path as osp
import os


class parser(object):
    def __init__(self):
        self.name = "training_cloth_segm_u2net_exp1_head"  # Expriment name
        self.image_folder = "C:\\Users\\19521\\OneDrive\\Desktop\\KHOALUAN\\cloth-segmentation\\output_images_split\\label_head"
        self.df_path = "C:\\Users\\19521\\OneDrive\\Desktop\\KHOALUAN\\cloth-segmentation\\split_data\\label_head.csv"  # label csv path
        self.distributed = False  # True for multi gpu training
        self.isTrain = True

        self.fine_width = 192 * 4
        self.fine_height = 192 * 4

        # Mean std params
        self.mean = 0.5
        self.std = 0.5

        self.batchSize = 12  # 12
        self.nThreads = 2  # 3
        self.max_dataset_size = float("inf")

        self.serial_batches = False
        self.continue_train = True
        if self.continue_train:
            self.unet_checkpoint = "prev_checkpoints/cloth_segm_unet_surgery.pth"

        self.save_freq = 1000
        self.print_freq = 10
        self.image_log_freq = 100

        self.iter = 1000
        self.lr = 0.0002
        self.clip_grad = 5

        self.logs_dir = osp.join("logs", self.name)
        self.save_dir = osp.join("results", self.name)
