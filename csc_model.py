import os
from copy import deepcopy

import numpy as np
import opencc
import pypinyin
import torch
from PIL import ImageFont
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MaskedLMOutput

from transformers import BertPreTrainedModel, BertModel


def _is_chinese_char(cp):
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False


class Pinyin2(object):
    def __init__(self):
        super(Pinyin2, self).__init__()
        pho_vocab = ['P']
        pho_vocab += [chr(x) for x in range(ord('1'), ord('5') + 1)]
        pho_vocab += [chr(x) for x in range(ord('a'), ord('z') + 1)]
        pho_vocab += ['U']
        assert len(pho_vocab) == 33
        self.pho_vocab_size = len(pho_vocab)
        self.pho_vocab = {c: idx for idx, c in enumerate(pho_vocab)}

    def get_pho_size(self):
        return self.pho_vocab_size

    @staticmethod
    def get_pinyin(c):
        if len(c) > 1:
            return 'U'
        s = pypinyin.pinyin(
            c,
            style=pypinyin.Style.TONE3,
            neutral_tone_with_five=True,
            errors=lambda x: ['U' for _ in x],
        )[0][0]
        if s == 'U':
            return s
        assert isinstance(s, str)
        assert s[-1] in '12345'
        s = s[-1] + s[:-1]
        return s

    def convert(self, chars):
        pinyins = list(map(self.get_pinyin, chars))
        pinyin_ids = [list(map(self.pho_vocab.get, pinyin)) for pinyin in pinyins]
        pinyin_lens = [len(pinyin) for pinyin in pinyins]
        pinyin_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in pinyin_ids],
            batch_first=True,
            padding_value=0,
        )
        return pinyin_ids, pinyin_lens


pho2_convertor = Pinyin2()


class CharResNet(torch.nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        # input_image: bxcx32x32, output_image: bx768x1x1
        self.res_block1 = BasicBlock(in_channels, 64, stride=2)  # channels: 64, size: 16x16
        self.res_block2 = BasicBlock(64, 128, stride=2)  # channels: 128, size: 8x8
        self.res_block3 = BasicBlock(128, 256, stride=2)  # channels: 256, size: 4x4
        self.res_block4 = BasicBlock(256, 512, stride=2)  # channels: 512, size: 2x2
        self.res_block5 = BasicBlock(512, 768, stride=2)  # channels: 768, size: 1x1

    def forward(self, x):
        # input_shape: bxcx32x32, output_image: bx768
        # x = x.unsqueeze(1)
        h = self.res_block1(x)
        h = self.res_block2(h)
        h = self.res_block3(h)
        h = self.res_block4(h)
        h = self.res_block5(h)
        h = h.squeeze(-1).squeeze(-1)
        return h


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class CharResNet1(torch.nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.res_block1 = BasicBlock(in_channels, 64, stride=2)  # channels: 64, size: 16x16
        self.res_block2 = BasicBlock(64, 128, stride=2)  # channels:  128, size: 8x8
        self.res_block3 = BasicBlock(128, 192, stride=2)  # channels: 256, size: 4x4
        self.res_block4 = BasicBlock(192, 192, stride=2)

    def forward(self, x):
        # input_shape: bxcx32x32, output_shape: bx128x8x8
        h = x
        h = self.res_block1(h)
        h = self.res_block2(h)
        h = self.res_block3(h)
        h = self.res_block4(h)
        h = h.view(h.shape[0], -1)
        return h


class ReaLiseForCSC(BertPreTrainedModel):

    def __init__(self, config):
        super(ReaLiseForCSC, self).__init__(config)
        self.config = config

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)

        self.pho_embeddings = nn.Embedding(pho2_convertor.get_pho_size(), config.hidden_size, padding_idx=0)
        self.pho_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        pho_config = deepcopy(config)
        pho_config.num_hidden_layers = 4
        self.pho_model = BertModel(pho_config)

        self.char_images_multifonts = torch.nn.Parameter(torch.rand(21128, 3, 32, 32))
        self.char_images_multifonts.requires_grad = False

        self.resnet = CharResNet(in_channels=3)
        self.resnet_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gate_net = nn.Linear(4 * config.hidden_size, 3)

        out_config = deepcopy(config)
        out_config.num_hidden_layers = 3
        self.output_block = BertModel(out_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

        self.loss_fnt = CrossEntropyLoss(ignore_index=0)

        self.tokenizer = None

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    def build_glyce_embed(self, vocab_dir, font_path, font_size=32):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [s.strip() for s in f]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) != 1 or (not _is_chinese_char(ord(char))):
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).reshape(char_images.shape[0], -1)
        assert char_images.shape == (21128, 1024)
        self.char_images.weight.data.copy_(char_images)

    # Add by hengdaxu
    def build_glyce_embed_multifonts(self, vocab_dir, num_fonts, use_traditional_font, font_size=32):
        font_paths = [
            ('simhei.ttf', False),
            ('xiaozhuan.ttf', False),
            ('simhei.ttf', True),
        ]
        font_paths = font_paths[:num_fonts]
        if use_traditional_font:
            font_paths = font_paths[:-1]
            font_paths.append(('simhei.ttf', True))
            self.converter = opencc.OpenCC('s2t.json')

        images_list = []
        for font_path, use_traditional in font_paths:
            images = self.build_glyce_embed_onefont(
                vocab_dir=vocab_dir,
                font_path=font_path,
                font_size=font_size,
                use_traditional=use_traditional,
            )
            images_list.append(images)

        char_images = torch.stack(images_list, dim=1).contiguous()
        self.char_images_multifonts.data.copy_(char_images)

    # Add by hengdaxu
    def build_glyce_embed_onefont(self, vocab_dir, font_path, font_size, use_traditional):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path, encoding='utf-8') as f:
            vocab = [s.strip() for s in f.readlines()]
        if use_traditional:
            vocab = [self.converter.convert(c) if len(c) == 1 else c for c in vocab]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) > 1:
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).contiguous()
        return char_images

    @staticmethod
    def build_batch(batch, tokenizer):
        src_idx = batch['src_idx'].flatten().tolist()
        chars = tokenizer.convert_ids_to_tokens(src_idx)
        pho_idx, pho_lens = pho2_convertor.convert(chars)
        batch['pho_idx'] = pho_idx
        batch['pho_lens'] = pho_lens
        return batch

    def forward(self,
                input_ids=None,
                pho_idx=None,
                pho_lens=None,
                attention_mask=None,
                labels=None,
                **kwargs):
        input_shape = input_ids.size()

        bert_hiddens = self.bert(input_ids, attention_mask=attention_mask)[0]

        pho_embeddings = self.pho_embeddings(pho_idx)
        pho_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=pho_embeddings,
            lengths=pho_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        _, pho_hiddens = self.pho_gru(pho_embeddings)
        pho_hiddens = pho_hiddens.squeeze(0).reshape(input_shape[0], input_shape[1], -1).contiguous()
        pho_hiddens = self.pho_model(inputs_embeds=pho_hiddens, attention_mask=attention_mask)[0]

        src_idxs = input_ids.view(-1)

        if self.config.num_fonts == 1:
            images = self.char_images(src_idxs).reshape(src_idxs.shape[0], 1, 32, 32).contiguous()
        else:
            images = self.char_images_multifonts.index_select(dim=0, index=src_idxs)

        res_hiddens = self.resnet(images)
        res_hiddens = res_hiddens.reshape(input_shape[0], input_shape[1], -1).contiguous()
        res_hiddens = self.resnet_layernorm(res_hiddens)

        bert_hiddens_mean = (bert_hiddens * attention_mask.to(torch.float).unsqueeze(2)).sum(dim=1) / attention_mask.to(
            torch.float).sum(dim=1, keepdim=True)
        bert_hiddens_mean = bert_hiddens_mean.unsqueeze(1).expand(-1, bert_hiddens.size(1), -1)

        concated_outputs = torch.cat((bert_hiddens, pho_hiddens, res_hiddens, bert_hiddens_mean), dim=-1)
        gated_values = self.gate_net(concated_outputs)
        # B * S * 3
        g0 = torch.sigmoid(gated_values[:, :, 0].unsqueeze(-1))
        g1 = torch.sigmoid(gated_values[:, :, 1].unsqueeze(-1))
        g2 = torch.sigmoid(gated_values[:, :, 2].unsqueeze(-1))

        hiddens = g0 * bert_hiddens + g1 * pho_hiddens + g2 * res_hiddens

        outputs = self.output_block(inputs_embeds=hiddens,
                                    position_ids=torch.zeros(input_ids.size(), dtype=torch.long,
                                                             device=input_ids.device),
                                    attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = MaskedLMOutput(
            logits=logits,
            hidden_states=outputs.last_hidden_state,
        )

        if labels is not None:
            # Only keep active parts of the loss
            labels[labels == 101] = 0
            labels[labels == 102] = 0
            loss = self.loss_fnt(logits.view(-1, logits.size(-1)), labels.view(-1))
            outputs.loss = loss

        return outputs

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def predict(self, sentences):
        if self.tokenizer is None:
            raise RuntimeError("Please init tokenizer by `set_tokenizer(tokenizer)` before predict.")

        str_flag = False
        if type(sentences) == str:
            sentences = [sentences]
            str_flag = True

        inputs = self.tokenizer(sentences, padding=True, return_tensors="pt")
        outputs = self.forward(**inputs).logits

        ids_list = outputs.argmax(-1)

        preds = []
        for i, ids in enumerate(ids_list):
            ids = ids[inputs['attention_mask'][i].bool()]
            pred_tokens = self.tokenizer.convert_ids_to_tokens(ids)
            pred_tokens = [t if not t.startswith('##') else t[2:] for t in pred_tokens]
            pred_tokens = [t if t != self.tokenizer.unk_token else '×' for t in pred_tokens]

            offsets = inputs[i].offsets
            src_tokens = list(sentences[i])
            for (start, end), pred_token in zip(offsets, pred_tokens):
                if end - start <= 0:
                    continue

                if (end - start) != len(pred_token):
                    continue

                if pred_token == '×':
                    continue

                if (end - start) == 1 and not _is_chinese_char(ord(src_tokens[start])):
                    continue

                src_tokens[start:end] = pred_token

            pred = ''.join(src_tokens)
            preds.append(pred)

        if str_flag:
            return preds[0]

        return preds
