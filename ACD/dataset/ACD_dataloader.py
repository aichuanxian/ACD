from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class ACDDataloader(object):
    def __init__(self, args, dataset, flag = 'train'):
        assert isinstance(dataset, Dataset)
        self.args = args
        self.dataset = dataset
        self.flag = flag
        self.batch_size = args.training.per_gpu_train_batch_size


    def get_dataloader(self):
        dataloader = DataLoader(dataset=self.dataset,
                                batch_size=self.batch_size,
                                shuffle=True if self.flag == True else False,
                                collate_fn=self.dataset.collate_fn)
        return dataloader
