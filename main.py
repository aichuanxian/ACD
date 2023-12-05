import torch
from utils.args_utils import Arguments
from ACD.trainer.ACD_trainer_base import ACDTrainer
from ACD.trainer.ACD_trainer_prompt import ACDPromptTrainer
from ACD.trainer.ACD_trainer_prefix import ACDPrefixTrainer
from scripts.checkpoint import avg_ckpt


if __name__ == '__main__':
    args = Arguments()

    print(f'torch.cuda.is_available:{torch.cuda.is_available()}')
    if torch.cuda.is_available():
        args.args.is_cuda = True

    if args.experiment.taskname == "ACD_base":
        my_trainer = ACDTrainer(args)
    elif args.experiment.taskname == "ACD_Prompt":
        my_trainer = ACDPromptTrainer(args)
    elif args.experiment.taskname == "ACD_Prefix":
        my_trainer = ACDPrefixTrainer(args)
        
    my_trainer.do_train()
    # avg_ckpt(args.experiment.input_ckpt_path, args.experiment.output_ckpt, args.experiment.num_epoch_checkpoints)
    my_trainer.do_test()
    
