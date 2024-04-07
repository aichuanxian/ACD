import time
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from transformers import AutoTokenizer

from utils.model_utils import save_model, load_model
from utils.metrics_utils import format_eval_output
from utils.plot_utils import use_svg_display,set_axes,set_figsize,plot, plot_attention_weights
from utils.log_utils import log_training_info

from ACD.trainer.train_base import TrainerBase
from ACD.models.ACD_model_with_prefix import ACDModelWithPrefix
from ACD.dataset.ACD_data_module import ACDDataModule


class ACDPrefixTrainer(TrainerBase):
    def __init__(self, args):
        super(ACDPrefixTrainer, self).__init__(args)
        self.args = args
        self.datamodule = ACDDataModule(args)
        self.models = ACDModelWithPrefix(args).cuda() if self.is_cuda else ACDModelWithPrefix(args)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path) 
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


    def train_epoch(self, epoch):
        self.models.train()
        train_loss, total_correct = 0, 0

        train_loader = self.get_train_dataloader()
        # layer_att_weights = []
        head0_weights = []
        for step, batch_data in tqdm(enumerate(train_loader)):
            batch_input_ids = batch_data['input_ids'].cuda() if self.is_cuda else batch_data['input_ids']
            batch_input_mask = batch_data['input_mask'].cuda() if self.is_cuda else batch_data['input_mask']
            batch_segment_ids = batch_data['segment_ids'].cuda() if self.is_cuda else batch_data['segment_ids']
            batch_label_ids = batch_data['label_ids'].cuda() if self.is_cuda else batch_data['label_ids']
            # print(f'batch_label_ids:{batch_label_ids},\nbatch_label_ids shape:{batch_label_ids.shape}')
            
            assert batch_input_mask.size(1) == batch_input_ids.size(1), "Mask length doesn't match input length"
            # print(f'batch_input_ids shape: {batch_input_ids.shape}\n\
            #         batch_input_mask shape: {batch_input_mask.shape}\n\
            #         batch_segment_ids shape: {batch_segment_ids.shape}\n\
            #         batch_label_ids shape: {batch_label_ids.shape}')
            loss, output, layer_att_weights = self.models(input_ids=batch_input_ids,
                                       attention_mask=batch_input_mask,
                                       token_type_ids=batch_segment_ids,
                                       labels=batch_label_ids,
                                       output_hidden_states=True)
            _, pred = torch.max(output, dim=-1)
            total_correct += torch.sum(pred == batch_label_ids).item()
            if step % self.args.training.optim.gradient_accumulation_steps > 1:
                loss = loss / self.args.training.optim.gradient_accumulation_steps
            train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(self.models.parameters(), max_norm=self.args.training.optim.max_grad_norm)
            # print('running')
            if step % self.args.training.optim.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            # Print the shape and content of layer_att_weights
            # for att_weight in layer_att_weights:
            #     print(att_weight[0].shape)
            
        avg_weights_head0 = [att_weight[0].mean(dim=[0, 1, 2]).detach().item() for att_weight in layer_att_weights]
        head0_weights.append(avg_weights_head0)

        print('Epoch {:2d} | Train Loss {:5.4f} |Train Acc {:5.4f}'.format(epoch,
                                                                      train_loss / len(train_loader),
                                                                      (total_correct / len(train_loader.dataset.inputs))*100))

        return train_loss / len(train_loader), (total_correct / len(train_loader.dataset.inputs))*100, layer_att_weights,head0_weights

    def do_evaluate(self, test_flag=False):
        self.models.eval()
        eval_loss, eval_correct = 0, 0
        rows = []
        counter = 0
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

                loss, output, layer_att_weights= self.models(input_ids=batch_input_ids,
                                           attention_mask=batch_input_mask,
                                           token_type_ids=batch_segment_ids,
                                           labels=batch_label_ids,
                                           output_attentions=True,
                                           output_hidden_states=True)
                eval_loss += loss.item()
                _, pred = torch.max(output, dim=-1)
                # print(f'prediction:{pred}')
                eval_correct += torch.sum(pred == batch_label_ids).item()

                rows.extend(
                    zip(
                        # batch_data["review_text"],
                        # batch_data["sentiment_targets"],
                        batch_data["label_ids"].numpy(),
                        pred.cpu().numpy(),
                    )
                )
                results = format_eval_output(rows)
                # counter += 1
                # if test_flag and counter < 5:
                #     # print(batch_data['input_ids'].shape)
                #     # print(batch_data['input_mask'].shape)
                #     masked_input_ids = torch.masked_select(batch_data['input_ids'], batch_data['input_mask'].bool())
                #     decoded_text = self.tokenizer.decode(masked_input_ids.tolist())
                #     print(f"Example {counter}")
                #     print(f"Input Text: {decoded_text}")
                #     print(f"Predicted Label: {results['prediction'].iloc[i]}")
                #     print(f"Actual Label: {results['label'].iloc[i]}\n")

        # print(results)
        self.models.train()

        print(f'eval_loss:{(eval_correct / len(dataloader.dataset.inputs))*100},  dataloader:{len(dataloader)}')
        
        return 1/(eval_loss / len(dataloader)), \
               (eval_correct / len(dataloader.dataset.inputs))*100, \
               format_eval_output(rows), \
                layer_att_weights

    def do_test(self):
        self.model = load_model(self.args)
        final_loss, final_acc, results, layer_att_weights = self.do_evaluate(test_flag=True)
        final_macro_f1 = f1_score(results.label, results.prediction, average="macro")*100
        
        plot_attention_weights(layer_att_weights, save_path=f'/root/02-ACD-Prompt_v1.0/attention_weights_test.png')
        
        print('Test acc {:5.4f} | Test Macro F1 {:5.4f}'.format(final_acc, final_macro_f1))
        print('testtttttttt')

        # for i in range(10):
        #     print(f"For example {i}, prediction: {results.prediction[i]}, actual label: {results.label[i]}")
        
        # 记录日志信息
        args_dict = {
        "Taskname": self.args.experiment.taskname,
        "Model_name":self.args.model.model_name_or_path,
        "With_parameter_freeze": self.args.experiment.with_parameter_freeze,
        "Train_data": self.args.data.text.train,
        "Hidden_dropout_prob": self.args.model.hidden_dropout_prob,
        "Pre_seq_len": self.args.model.pre_seq_len,
        # "Max_seq_length": self.args.model.max_seq_length,
        "Epochs": self.args.training.epochs,
        "Patience": self.args.training.patience,
        "Per_gpu_train_batch_size": self.args.training.per_gpu_train_batch_size,
        "Per_gpu_eval_batch_size": self.args.training.per_gpu_eval_batch_size,
        "Learning_rate": self.args.training.optim.learning_rate,
        "prefix_hidden_size": self.args.model.prefix_hidden_size               
        }

        log_training_info(args_dict, final_acc, final_macro_f1, final_loss, log_dir=self.args.args.log_save_path)

    def do_train(self):
        assert self.optimizer is not None
        patience = self.patience
        train_losses, train_accs, eval_losses, eval_accs, result, all_head0_weights = [], [], [], [], [], []

        for epoch in range(self.args.training.epochs):
            start = time.time()
            train_loss, train_acc, layer_att_weights, head0_weights= self.train_epoch(epoch)
            eval_loss, eval_acc, results, _ = self.do_evaluate()

            all_head0_weights.append(head0_weights)  # collect head0 weights

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            eval_losses.append(eval_loss)
            eval_accs.append(eval_acc)
            result.append(results)
            
            duration = time.time() - start

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
                print(f"Early Break!")
                break
            plot_attention_weights(layer_att_weights, save_path=f'/root/02-ACD-Prompt_v1.0/attention_weights_epoch_{epoch}.png')
                        # Plot the weights of head 0 for all epochs
            
            head0_weights_matrix = np.vstack(all_head0_weights).T  # transpose to make layers as rows
            plt.figure(figsize=(12, 8))
            sns.heatmap(head0_weights_matrix, annot=True, fmt=".2f")
            plt.xlabel('Epoch')
            plt.ylabel('Layer')
            plt.savefig(f'/root/02-ACD-Prompt_v1.0/head0_weights_epoch_{epoch}.png')
            plt.close()

        self.model = load_model(self.args)
        final_loss, final_acc, results, _ = self.do_evaluate(test_flag=True)
        # print(results)
        final_macro_f1 = f1_score(
            results.label, results.prediction, average="macro"
        )
        print('Final acc {:5.4f} | Final Macro F1 {:5.4f} | Final Loss {:5.4f}'.format(final_acc, final_macro_f1, final_loss))
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
