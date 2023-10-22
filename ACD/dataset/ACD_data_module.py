from ACD.dataset.ACD_dataset import ACDDataset
from ACD.dataset.ACD_dataloader import ACDDataloader
from transformers import BertTokenizer
import torch 


class ACDDataModule(object):
    def __init__(self, args):
        self.args = args
        self.num_train_steps = -1

        self.datasets = {}
        self.dataset_to_iter = {}
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=args.model.model_name_or_path,
                                                       do_lower_case=args.model.do_lower_case)

    def dataset(self, flag = 'train'):
        return self.datasets.get(flag, None)

    # def load_dataset(self, flag='train'):
    #     dataset = ACDDataset(self.args, self.tokenizer, flag)
        
    #     self.datasets[flag] = dataset.truncate(max_length=self.args.model.max_seq_length)

    #     if flag == 'train':
    #         self.num_train_steps = int(len(self.datasets[flag].all_token_ids)
    #                                    /self.args.batch_size
    #                                    / self.args.gradient_accumulation_steps
    #                                    * self.args.num_epoch)

    def load_dataset(self, flag = 'train'):
        dataset = ACDDataset(self.args, self.tokenizer, flag)
        # self.datasets[flag] = dataset.truncate(max_length=self.args.model.max_seq_length)
        self.datasets[flag] = dataset

        if flag == 'train':
            self.num_train_steps = int(len(self.datasets[flag].all_token_ids)
                                       /self.args.training.per_gpu_train_batch_size
                                       / self.args.training.optim.gradient_accumulation_steps
                                       * self.args.training.epochs)
                                      
    def get_dataloader(self, dataset):
        if dataset in self.dataset_to_iter:
            return self.dataset_to_iter[dataset]
        else:
            assert isinstance(dataset, ACDDataset)
            dataloader = ACDDataloader(args=self.args,
                                             dataset=dataset,
                                             flag=dataset.flag).get_dataloader()
            self.dataset_to_iter[dataset] = dataloader
            return dataloader


