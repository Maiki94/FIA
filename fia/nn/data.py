import os
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from fia.constants import COL_DF_LABEL_FRAUD


class DataFrameDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, label_column: str):
        """
        Args:
            dataframe (pd.DataFrame): Input data in pandas DataFrame format.
            label_column (str): Name of the column to be used as the labels.
        """

        self.features = dataframe.drop(label_column, axis=1).values
        self.labels = dataframe[label_column].values

        # Convert features and labels to torch tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DataModule(LightningDataModule):
    batch_size: int
    random_sate: int
    persistent_workers: int
    num_workers: int

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        batch_size: int = 32,
        random_sate: int = 42,
        num_workers: int = int(os.cpu_count()*0.8),
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
    ):
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_sate
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        self._class_weights = None
        self.is_data_splitted = False

    def _create_tensor_dataset_from_dataframe(self, dataframe: pd.DataFrame):
        return DataFrameDataset(dataframe, COL_DF_LABEL_FRAUD)

    def setup(self, stage: str):

        if stage == "fit" or stage is None:
            self.train_dataset = self._create_tensor_dataset_from_dataframe(
                self.train_df
            )
            self.val_dataset = self._create_tensor_dataset_from_dataframe(self.val_df)

        if stage == "test" or stage is None:
            self.test_dataset = self._create_tensor_dataset_from_dataframe(self.test_df)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            batch_size=self.batch_size,
            prefetch_factor=self.prefetch_factor,
            dataset=self.train_dataset,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
        )


# import os
# import pandas as pd
# import torch
# from torch import Tensor
# from torch.utils.data import DataLoader, Dataset
# from pytorch_lightning import LightningDataModule

# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight

# from fia.constants import COL_DF_LABEL_FRAUD


# class DataFrameDataset(Dataset):
#     def __init__(self, dataframe: pd.DataFrame, label_column: str):
#         """
#         Args:
#             dataframe (pd.DataFrame): Input data in pandas DataFrame format.
#             label_column (str): Name of the column to be used as the labels.
#         """

#         self.features = dataframe.drop(label_column, axis=1).values
#         self.labels = dataframe[label_column].values

#         # Convert features and labels to torch tensors
#         self.features = torch.tensor(self.features, dtype=torch.float32)
#         self.labels = torch.tensor(self.labels, dtype=torch.float32)

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]


# class DataModule(LightningDataModule):
#     dataframe: pd.DataFrame
#     batch_size: int
#     random_sate: int
#     persistent_workers: int
#     num_workers: int

#     def __init__(
#         self,
#         dataframe: pd.DataFrame,
#         batch_size: int = 32,
#         test_size: float = 0.15,
#         val_size: float = 0.15,
#         random_sate: int = 42,
#         num_workers: int = os.cpu_count(),
#         persistent_workers: bool = True,
#         prefetch_factor: int = 2,
#     ):
#         super().__init__()

#         self.dataframe = dataframe
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.test_size = test_size
#         self.val_size = val_size
#         self.random_state = random_sate
#         self.persistent_workers = persistent_workers
#         self.prefetch_factor = prefetch_factor

#         self._class_weights = None
#         self.is_data_splitted = False

#     @property
#     def class_weights(self) -> Tensor:
#         if self._class_weights is None:
#             if not self.is_data_splitted:
#                 self._split_data()
#             self._class_weights = self._get_class_weights(self.train_df)
#         return self._class_weights

#     def _split_data(self):
#         if not self.is_data_splitted:
#             # Split to create a test dataframe
#             train_df, test_df = train_test_split(
#                 self.dataframe,
#                 test_size=self.test_size,
#                 random_state=self.random_state,
#                 shuffle=True,
#                 stratify=self.dataframe[COL_DF_LABEL_FRAUD],
#             )

#             # Split to create a train and validation dataframe
#             train_df, val_df = train_test_split(
#                 train_df,
#                 test_size=self.val_size,
#                 shuffle=True,
#                 random_state=self.random_state,
#                 stratify=train_df[COL_DF_LABEL_FRAUD],
#             )

#             # Store the splits for later stages
#             self.train_df = train_df
#             self.val_df = val_df
#             self.test_df = test_df

#             # Mark that the split has been done
#             self.data_split_done = True
#             # del self.dataframe

#     def _get_class_weights(self, dataframe: pd.DataFrame) -> Tensor:
#         class_weights = compute_class_weight(
#             class_weight="balanced",
#             classes=self.train_df[COL_DF_LABEL_FRAUD].unique(),
#             y=self.train_df[COL_DF_LABEL_FRAUD],
#         )
#         return torch.tensor(data=class_weights, dtype=torch.float32)

#     def _create_tensor_dataset_from_dataframe(self, dataframe: pd.DataFrame):
#         return DataFrameDataset(dataframe, COL_DF_LABEL_FRAUD)

#     def setup(self, stage: str):
#         if not self.is_data_splitted:
#             self._split_data()

#         if stage == "fit" or stage is None:
#             self.train_dataset = self._create_tensor_dataset_from_dataframe(
#                 self.train_df
#             )
#             self.val_dataset = self._create_tensor_dataset_from_dataframe(self.val_df)

#         if stage == "test" or stage is None:
#             self.test_dataset = self._create_tensor_dataset_from_dataframe(self.test_df)

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             shuffle=True,
#             num_workers=self.num_workers,
#             persistent_workers=self.persistent_workers,
#             batch_size=self.batch_size,
#             prefetch_factor=self.prefetch_factor,
#             dataset=self.train_dataset,
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             dataset=self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             persistent_workers=self.persistent_workers,
#             num_workers=self.num_workers,
#         )

#     def test_dataloader(self) -> DataLoader:
#         return DataLoader(
#             dataset=self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             persistent_workers=self.persistent_workers,
#             num_workers=self.num_workers,
#         )
