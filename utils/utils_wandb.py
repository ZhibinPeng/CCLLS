# import
import wandb
from torch.utils.tensorboard import SummaryWriter


class Config(object):
    def __init__(self, logger_name="wandb"):
        self.project_name = "casee"
        self.logger_name = logger_name
        self.log_dir = "./data/log/"


class Logger(object):
    def __init__(self, config):
        self.config = config
        self.logger_name = self.config.logger_name
        self.logger = self._get_logger()

    def _get_logger(self):
        logger = None

        if self.logger_name == "wandb":
            wandb.init(project=self.config.project_name,
                       config=self.config.__dict__)
            logger = wandb

        elif self.logger_name == "tensorboard":
            logger = SummaryWriter(self.config.log_dir)

        return logger

    def log(self, info):
        if self.logger_name == "wandb":
            self.logger.log(info)
        elif self.logger_name == "tensorboard":
            main_tag, tag_scalar_dict, global_step = info
            self.logger.add_scalars(main_tag, tag_scalar_dict, global_step)