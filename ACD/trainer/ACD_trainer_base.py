import time

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score

from ACD.trainer.train_base import TrainerBase
from utils.model_utils import save_model, load_model
from utils.metrics_utils import format_eval_output
from utils.plot_utils import use_svg_display,set_axes,set_figsize,plot
from utils.log_utils import log_training_info
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import numpy as np

from ACD.models.ACD_model import ACDModel
from ACD.dataset.ACD_data_module import ACDDataModule


class ACDTrainer(TrainerBase):
    def __init__(self, args):
        super(ACDTrainer, self).__init__(args)
        self.args = args
        self.datamodule = ACDDataModule(args)
        self.models = ACDModel(args).cuda() if self.is_cuda else ACDModel(args)
        self._initial_optimizer_schedule()

    def _initial_optimizer_schedule(self):
        if self.datamodule.dataset(flag='train') is None:
            self.datamodule.load_dataset(flag='train')

        # Prepare optimizer
        params = dict(self.models.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        paras_new = []
        for k, v in params.items():
            if 'bert' in k:
                if not any(nd in k for nd in no_decay):
                    paras_new += [{'params': [v], 'lr': self.args.training.optim.learning_rate, 'weight_decay': 0.01}]
                if any(nd in k for nd in no_decay):
                    paras_new += [{'params': [v], 'lr': self.args.training.optim.learning_rate, 'weight_decay': 0.0}]
            else:
                paras_new += [{'params': [v], 'lr': self.args.training.optim.learning_rate, 'weight_decay': 0.01}]

        # self.optimizer = optim.AdamW(paras_new, correct_bias=False)
        self.optimizer = optim.AdamW(paras_new)

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.args.training.optim.warmup_steps,
                                                         num_training_steps=self.datamodule.num_train_steps)

    def _init_optimizer(self):
        self.optimizer = AdamW(self.models.parameters(), lr=self.args.lr)

    def train_epoch(self, epoch):
        self.models.train()
        train_loss, total_correct = 0, 0

        train_loader = self.get_train_dataloader()
        for step, batch_data in tqdm(enumerate(train_loader)):
            batch_input_ids = batch_data['input_ids'].cuda() if self.is_cuda else batch_data['input_ids']
            batch_input_mask = batch_data['input_mask'].cuda() if self.is_cuda else batch_data['input_mask']
            batch_segment_ids = batch_data['segment_ids'].cuda() if self.is_cuda else batch_data['segment_ids']
            batch_label_ids = batch_data['label_ids'].cuda() if self.is_cuda else batch_data['label_ids']
            #print(f'label_shape:{batch_label_ids.shape}')
            loss, output = self.models(input_ids=batch_input_ids,
                                       input_mask=batch_input_mask,
                                       segment_ids=batch_segment_ids,
                                       label=batch_label_ids)

            _, pred = torch.max(output, dim=-1)
            total_correct += torch.sum(pred == batch_label_ids).item()
            if step % self.args.training.optim.gradient_accumulation_steps > 1:
                loss = loss / self.args.training.optim.gradient_accumulation_steps
            train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(self.models.parameters(), max_norm=self.args.training.optim.max_grad_norm)
            if step % self.args.training.optim.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        print('Epoch {:2d} | Train Loss {:5.4f} |Train Acc {:5.4f}'.format(epoch,
                                                                      train_loss / len(train_loader),
                                                                      (total_correct / len(train_loader.dataset.inputs))*100))

        return train_loss / len(train_loader), (total_correct / len(train_loader.dataset.inputs))*100

    def do_evaluate(self, test_flag=False):
        self.models.eval()
        eval_loss, eval_correct = 0, 0
        rows = []
        if test_flag:
            dataloader = self.get_test_dataloader()
        else:
            dataloader = self.get_valid_dataloader()

        with torch.no_grad():
            for batch_data in tqdm(dataloader):
                batch_input_ids = batch_data['input_ids'].cuda() if self.is_cuda else batch_data['input_ids']
                batch_input_mask = batch_data['input_mask'].cuda() if self.is_cuda else batch_data['input_mask']
                batch_segment_ids = batch_data['segment_ids'].cuda() if self.is_cuda else batch_data['segment_ids']
                batch_label_ids = batch_data['label_ids'].cuda() if self.is_cuda else batch_data['label_ids']

                loss, output = self.models(input_ids=batch_input_ids,
                                           input_mask=batch_input_mask,
                                           segment_ids=batch_segment_ids,
                                           label=batch_label_ids)
                eval_loss += loss.item()
                _, pred = torch.max(output, dim=-1)
                eval_correct += torch.sum(pred == batch_label_ids).item()

                rows.extend(
                    zip(
                        # batch_data["review_text"],
                        # batch_data["sentiment_targets"],
                        batch_data["label_ids"].numpy(),
                        pred.cpu().numpy(),
                    )
                )

        # print(rows)
        self.models.train()

        return eval_loss / len(dataloader), \
               (eval_correct / len(dataloader.dataset.inputs))*100, \
               format_eval_output(rows)

    def do_test(self):
        self.model = load_model(self.args)
        final_loss, final_acc, results = self.do_evaluate(test_flag=True)
        final_macro_f1 = f1_score(
            results.label, results.prediction, average="macro"
        )*100
        print('Test acc {:5.4f} | Test Macro F1 {:5.4f}'.format(final_acc, final_macro_f1))
        print('testtttttttt')

        # 记录日志信息
        args_dict = {
        "Taskname": self.args.experiment.taskname,
        "Model_name":self.args.model.model_name_or_path,
        "With_parameter_freeze": self.args.experiment.with_parameter_freeze,
        "Train_data": self.args.data.text.train,
        "Hidden_dropout_prob": self.args.model.hidden_dropout_prob,
        "Pre_seq_len": self.args.model.pre_seq_len,
        "Max_seq_length": self.args.model.max_seq_length,
        "Epochs": self.args.training.epochs,
        "Patience": self.args.training.patience,
        "Per_gpu_train_batch_size": self.args.training.per_gpu_train_batch_size,
        "Per_gpu_eval_batch_size": self.args.training.per_gpu_eval_batch_size,
        "Learning_rate": self.args.training.optim.learning_rate,                  
        }

        log_training_info(args_dict, final_acc, final_macro_f1, final_loss, log_dir=self.args.args.log_save_path)

    def do_train(self):
        assert self.optimizer is not None
        patience = self.patience
        train_losses, train_accs, eval_losses, eval_accs, result =[], [], [], [], []

        for epoch in range(self.args.training.epochs):
            start = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            eval_loss, eval_acc, results = self.do_evaluate()

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            eval_losses.append(eval_loss)
            eval_accs.append(eval_acc)
            result.append(results)

            duration = time.time() - start
            model_name = 'model_' + str(epoch + 1)
            #save_model(self.models, self.args.model.model_save_path, model_name)

            print('=' * 50)
            print('Epoch {:2d} | '
                  'Time {:5.4f} sec | '
                  'Train Loss {:5.4f} | '
                  'Valid Loss {:5.4f} | '
                  'Valid ACC {:5.4f} '.format(epoch, duration,
                                              train_loss, eval_loss, eval_acc))
            print("-" * 50)

            if eval_acc > self.best_valid_acc:
                patience = self.args.training.patience
                self.best_valid_acc = eval_acc
                save_model(self.models, self.args.model.model_save_path)
            else:
                patience -= 1
                print(patience)

            if patience <= 0:
                print(f"Early stopping")
                
                break

        # self.model = load_model(self.args.model.model_save_path)
        self.model = load_model(self.args)
        final_loss, final_acc, results = self.do_evaluate(test_flag=True)
        final_macro_f1 = f1_score(
            results.label, results.prediction, average="macro"
        )*100
        print('Final acc {:5.4f} | Final Macro F1 {:5.4f}'.format(final_acc, final_macro_f1))         
        print('Validdddddd')

        plot(np.arange(0, epoch + 1, 1), \
            [ train_accs, eval_accs], \
            xlabel='Epoch', \
            ylabel='Score',\
            legend=[ 'train_acc',  'eval_acc'],\
            figsize=(5.5, 4),\
            svg_save_path = '/root/02-ACD-Prompt_v1.0/plot-acc.png')
        
        plot(np.arange(0, epoch + 1, 1), \
            [train_losses,  eval_losses], \
            xlabel='Epoch', \
            ylabel='Score',\
            legend=['train_loss', 'eval_loss'],\
            figsize=(5.5, 4),\
            svg_save_path = '/root/02-ACD-Prompt_v1.0/plot-loss.png')