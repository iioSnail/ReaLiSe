import os

import torch

from csc_tokenizer import ReaLiSeTokenizer
from csc_model import ReaLiseForCSC
from transformers import BertConfig

output_path = "../iioSnail/ReaLiSe-for-csc"
os.makedirs(output_path, exist_ok=True)


def load_tokenizer():
    tokenizer = ReaLiSeTokenizer.from_pretrained("../output")
    return tokenizer


def load_model():
    config = {
        "attention_probs_dropout_prob": 0.1,
        "directionality": "bidi",
        "finetuning_task": None,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "image_model_type": 0,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "is_decoder": False,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_fonts": 3,
        "num_hidden_layers": 12,
        "num_labels": 2,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_past": True,
        "pooler_fc_size": 768,
        "pooler_num_attention_heads": 12,
        "pooler_num_fc_layers": 3,
        "pooler_size_per_head": 128,
        "pooler_type": "first_token_transform",
        "pruned_heads": {},
        "torchscript": False,
        "type_vocab_size": 2,
        "use_bfloat16": False,
        "vocab_size": 21128,
        "vocab_size_or_config_json_file": 21128
    }

    config = BertConfig(**config)

    model = ReaLiseForCSC(config)
    state_dict = torch.load("../output/pytorch_model.bin", map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def model_test(tokenizer, model):
    inputs = tokenizer(["我喜欢吃平果", "她喜欢持梨"], padding=True, return_tensors="pt")
    outputs = model(**inputs)

    print(tokenizer.convert_ids_to_tokens(outputs.logits[0].argmax(-1)))

    inputs = tokenizer(["我喜欢吃平果", "她喜欢持梨"], text_target=["我喜欢吃苹果", "她喜欢吃梨"], padding=True, return_tensors="pt")
    outputs = model(**inputs)
    print(outputs.loss)

    inputs = tokenizer("我喜欢吃平果", text_target="我喜欢吃苹果", return_tensors="pt")
    inputs = inputs.to('cpu')
    outputs = model(**inputs)
    print(outputs.loss)

    tokenizer(["我喜欢吃平果", "她喜欢持梨"])

    tokenizer(["我喜欢吃平果", "她喜欢持梨"], padding=True)

    tokenizer("我喜欢吃平果")

    model.set_tokenizer(tokenizer)

    model.predict("我喜欢吃 #平果，一次能吃34 个 %%！xx×")

    model.predict(["我喜欢吃 #平果，一次能吃34 个 %%！xx×", "吃 #平果，一次能吃34 个 %%！xx×"])

    print()


def export_tokenizer(tokenizer):
    tokenizer.register_for_auto_class("AutoTokenizer")
    tokenizer.save_vocabulary(output_path)
    tokenizer.save_pretrained(output_path)


def export_model(model):
    model.register_for_auto_class("AutoModel")
    model.save_pretrained(output_path)


def main():
    tokenizer = load_tokenizer()
    model = load_model()

    model_test(tokenizer, model)

    export_tokenizer(tokenizer)
    export_model(model)

    print("Export success!")


if __name__ == '__main__':
    main()
    # import transformers
    # print(transformers.__version__)
