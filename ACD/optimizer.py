from transformers import AdamW, get_linear_schedule_with_warmup

def get_optimizer_and_scheduler(args, model, num_training_sampels):
    oargs = args.optim

    if oargs.max_steps > 0:
        t_total = oargs.max_steps
        oargs.num_train_epochs = oargs.max_steps // (num_training_sampels // oargs.gradient_accumulation_steps) + 1
    else:
        t_total = num_training_sampels // oargs.gradient_accumulation_steps * args.training.epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": oargs.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=oargs.learning_rate,
                      betas=(oargs.beta_1, oargs.beta_2),
                      eps=oargs.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=oargs.warmup_steps,
                                                num_training_steps=t_total)

    return optimizer, scheduler


