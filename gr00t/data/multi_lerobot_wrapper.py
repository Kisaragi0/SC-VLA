import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP

import bisect

class MultiDatasetWrapper(Dataset):
    def __init__(self, datasets, sampling_weights):
        """
        Args:
            datasets: list of LeRobotSingleDataset instances
            sampling_weights: list of float, same length as datasets, should sum to 1
        """
        # assert len(datasets) == len(sampling_weights)
        self.datasets = datasets
        self.weights = sampling_weights
        self.dataset_lengths = [len(d) for d in datasets]
        self.cum_lengths = self._compute_cum_lengths(self.dataset_lengths)  
        self.total_length = sum(self.dataset_lengths)

    def _compute_cum_lengths(self, lengths):
        cum = [0]
        for l in lengths:
            cum.append(cum[-1] + l)
        return cum

    def __len__(self):
        # You can set this arbitrarily or to a multiple of epoch size
        return self.total_length

    def __getitem__(self, idx):
        # Randomly choose a dataset based on weights
        dataset_idx = bisect.bisect_right(self.cum_lengths, idx) - 1
        local_idx = idx - self.cum_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]

if __name__ == '__main__':

    dataset_path = ['', '']
    embodiment_tag = ['new_embodiment','libero']
    data_config = ['so100', 'libero']
    video_backend = ['torchvision_av','torchvision_av']
    for config in zip(dataset_path, embodiment_tag, data_config, video_backend):
        print(config)

    train_dataset = [LeRobotSingleDataset(
        dataset_path=config[0],
        modality_configs=DATA_CONFIG_MAP[config[2]].modality_config(),
        transforms=DATA_CONFIG_MAP[config[2]].transform(),
        embodiment_tag=EmbodimentTag(config[1]),  # This will override the dataset's embodiment tag to "new_embodiment"
        video_backend=config[3],
    ) for config in zip(dataset_path, embodiment_tag, data_config, video_backend)]

    multi_dataset = MultiDatasetWrapper(
        datasets=train_dataset,  
        sampling_weights=[0.4,0.6],
    )
    loader = DataLoader(multi_dataset, batch_size=1, num_workers=4)
    for batch in loader:
        print(batch)

        
    single_dataset = LeRobotSingleDataset(
        dataset_path=dataset_path[1],
        modality_configs=DATA_CONFIG_MAP[data_config[1]].modality_config(),
        transforms=DATA_CONFIG_MAP[data_config[1]].transform(),
        embodiment_tag=EmbodimentTag(embodiment_tag[1]),  # This will override the dataset's embodiment tag to "new_embodiment"
        video_backend=video_backend[1]
    )
    loader = DataLoader(single_dataset, batch_size=16, num_workers=4, shuffle=True)
    for batch in loader:
        print(batch)
    