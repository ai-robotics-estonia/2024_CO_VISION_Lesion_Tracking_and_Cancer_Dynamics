import hydra
from omegaconf import OmegaConf

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: OmegaConf,
    ) -> None:
        super().__init__()

        self.config = config

    def setup(self, stage: str) -> None:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return hydra.utils.instantiate(
            self.config.dataloaders.train_dataloader, dataset=self.train_data
        )

    def val_dataloader(self) -> DataLoader:
        return hydra.utils.instantiate(
            self.config.dataloaders.val_dataloader, dataset=self.val_data
        )

    def test_dataloader(self) -> DataLoader:
        return hydra.utils.instantiate(
            self.config.dataloaders.test_dataloader, dataset=self.test_data
        )
