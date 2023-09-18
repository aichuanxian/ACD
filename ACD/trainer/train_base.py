

class TrainerBase(object):
    def __init__(self, args):
        self.args = args
        self.datamodule = None
        self.models = None
        self._init_parameters()

    def _init_parameters(self):
        self.is_cuda = self.args.args.is_cuda
        self.patience = self.args.args.patience
        self.best_valid_loss = 1e8
        self.best_valid_acc = 0

    def get_train_dataloader(self):
        if self.datamodule.dataset(flag='train') is None:
            self.datamodule.load_dataset(flag='train')
        return self.datamodule.get_dataloader(dataset=self.datamodule.dataset(flag='train'))

    def get_valid_dataloader(self):
        if self.datamodule.dataset(flag='valid') is None:
            self.datamodule.load_dataset(flag='valid')
        return self.datamodule.get_dataloader(dataset=self.datamodule.dataset(flag='valid'))

    def get_test_dataloader(self):
        if self.datamodule.dataset(flag='test') is None:
            self.datamodule.load_dataset(flag='test')
        return self.datamodule.get_dataloader(dataset=self.datamodule.dataset(flag='test'))

    def do_train(self):
        raise NotImplementedError
