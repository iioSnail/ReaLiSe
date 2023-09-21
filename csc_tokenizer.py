from typing import List, Union, Optional

import pypinyin
import torch
from torch import NoneType

from transformers import BertTokenizerFast


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


class ReaLiSeTokenizer(BertTokenizerFast):

    def __init__(self, **kwargs):
        super(ReaLiSeTokenizer, self).__init__(**kwargs)

        self.pho2_convertor = Pinyin2()

    def __call__(self,
                 text: Union[str, List[str], List[List[str]]] = None,
                 text_pair: Union[str, List[str], List[List[str]], NoneType] = None,
                 text_target: Union[str, List[str], List[List[str]]] = None,
                 text_pair_target: Union[str, List[str], List[List[str]], NoneType] = None,
                 add_special_tokens: bool = True,
                 padding=False,
                 truncation=None,
                 max_length: Optional[int] = None,
                 stride: int = 0,
                 is_split_into_words: bool = False,
                 pad_to_multiple_of: Optional[int] = None,
                 return_tensors=None,
                 return_token_type_ids: Optional[bool] = None,
                 return_attention_mask: Optional[bool] = None,
                 return_overflowing_tokens: bool = False, return_special_tokens_mask: bool = False,
                 return_offsets_mapping: bool = False,
                 return_length: bool = False,
                 verbose: bool = True, **kwargs):
        encoding = super(ReaLiSeTokenizer, self).__call__(
            text=text,
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
        )

        input_ids = encoding['input_ids']
        if type(text) == str and return_tensors is None:
            input_ids = [input_ids]

        pho_idx_list = []
        pho_lens_list = []
        for ids in input_ids:
            chars = self.convert_ids_to_tokens(ids)
            pho_idx, pho_lens = self.pho2_convertor.convert(chars)
            if return_tensors is None:
                pho_idx = pho_idx.tolist()
            pho_idx_list.append(pho_idx)
            pho_lens_list += pho_lens

        pho_idx = pho_idx_list
        pho_lens = pho_lens_list
        if return_tensors == 'pt':
            pho_idx = torch.vstack(pho_idx)
            pho_lens = torch.LongTensor(pho_lens)

        if type(text) == str and return_tensors is None:
            pho_idx = pho_idx[0]

        encoding['pho_idx'] = pho_idx
        encoding['pho_lens'] = pho_lens

        return encoding
