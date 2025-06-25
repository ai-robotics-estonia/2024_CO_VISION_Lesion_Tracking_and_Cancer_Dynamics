from itertools import chain
from omegaconf import OmegaConf
from hydra.utils import instantiate
import time

from monai.data import CacheDataset, Dataset
from lightning_modules.datamodules.base_datamodule import BaseDataModule

from dataset.registration.utils import get_files

import hydra

class RegistrationDataModule(BaseDataModule):
    def __init__(
        self,
        config: OmegaConf,
    ) -> None:
        super().__init__(config)


    def setup(self, stage: str) -> None:
        
        if stage == "fit":

            train_datalist, val_datalist = get_files(self.config.data_dir)
            # train_datalist = train_datalist[:100]
            # val_datalist = val_datalist[:100]

            train_transforms = instantiate(self.config.transforms.train_transforms)
            val_transforms = instantiate(self.config.transforms.val_transforms)
            start_time = time.time()
            print("Caching training data...")

            # self.train_data = CacheDataset(
            #     data=train_datalist,
            #     transform=train_transforms,
            #     cache_rate=self.config.cache_rate,
            #     num_workers=self.config.num_workers,
            # )
            # self.val_data = CacheDataset(
            #     data=val_datalist,
            #     transform=val_transforms,
            #     cache_rate=self.config.cache_rate,
            #     num_workers=self.config.num_workers,
            # )

            self.train_data = Dataset(
                data=train_datalist,
                transform=val_transforms,
            )
            self.val_data = Dataset(
                data=val_datalist,
                transform=val_transforms,
            )

        

            finish_time = time.time()
            print(f"Finished caching data in {finish_time - start_time:.2f} seconds")


        elif stage == "test":
            # TODO: implement test dataloader
            # test_datalist = instantiate(self.config.test_datalist)
            # test_transforms = instantiate(self.config.transforms.test_transforms)

            # self.test_data = Dataset(
            #     data=test_datalist,
            #     transform=test_transforms,
            # )
            pass