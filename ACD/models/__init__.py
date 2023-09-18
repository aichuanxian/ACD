from ACD.models.ACD_model import ACDModel
from ACD.models.ACD_model_with_prompt import ACDModelWithPrompt
from transformers import BertConfig, AutoConfig


def build_model(args):
    model_name = args.model.name
    config = prepare_model_config(args)

    # if model_name == 'ner':
    #     print('Unsupport model...')
    #     model_class = NERModel(config)
    # elif model_name == 'ner_with_caption':
    #     model_class = NERWithCaption(config)
    # elif model_name == 'mner':
    if model_name == 'ACD':
        # model_class = VisualBertModel(config)
        model_class = ACDModel(args, config)
    elif model_name == 'ACD_prompt':
        model_class = ACDModelWithPrompt(args, config)
    else:
        raise NotImplementedError

    return model_class


def prepare_model_config(args):
    config = BertConfig.from_pretrained(args.model.model_name_or_path)
    config.pre_seq_len = args.model.pre_seq_len

    config.model_name_or_path = args.model.model_name_or_path
    config.num_labels = len(args.data.label_scheme)
    config.output_attentions = args.model.output_attentions
    config.output_hidden_states = args.model.output_hidden_states
    config.visual_embedding_dim = args.data.image.embedding_dim
    config.visual_embedding_size = args.data.image.size
    config.ckpt_path = args.model.pretrained_weights

    return config