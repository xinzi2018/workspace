from parameter import *
from trainer import Trainer
from torch.backends import cudnn
from utils import make_folder
import os
import torch
from cartoon_hander import MyDataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL.Image import BILINEAR

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Data loader
    data_loader = torch.utils.data.DataLoader(
        MyDataset(config.image_path, img_transform=Compose([
            Resize(224, interpolation=BILINEAR),
            # CenterCrop(500),
            ToTensor()
        ])),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,

    )
    test_data = torch.utils.data.DataLoader(
        MyDataset(config.test_data_path, img_transform=Compose([
            Resize(224, interpolation=BILINEAR),
            # CenterCrop(500),
            ToTensor()
        ])),
        batch_size=1,
        shuffle=True,
        num_workers=2,

    )

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)


    if config.train:
        if config.model == 'sagan':
            trainer = Trainer(data_loader, test_data, config)
        # elif config.model == 'qgan':
        #     trainer = qgan_trainer(data_loader.loader(), config)
        trainer.train()
    # else:
    #     tester = Tester(data_loader.loader(), config)
    #     tester.test()


if __name__ == '__main__':
    config = get_parameters()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.myGpu
    print(config)
    main(config)