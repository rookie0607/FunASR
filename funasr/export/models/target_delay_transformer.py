from typing import Any
from typing import List
from typing import Tuple

import torch
import torch.nn as nn

from funasr.export.utils.torch_function import MakePadMask
from funasr.export.utils.torch_function import sequence_mask
#from funasr.models.encoder.sanm_encoder import SANMEncoder as Encoder
from funasr.punctuation.sanm_encoder import SANMEncoder
from funasr.export.models.encoder.sanm_encoder import SANMEncoder as SANMEncoder_export
from funasr.punctuation.abs_model import AbsPunctuation


class TargetDelayTransformer(nn.Module):

    def __init__(
            self,
            model,
            max_seq_len=512,
            model_name='punc_model',
            **kwargs,
    ):
        super().__init__()
        onnx = False
        if "onnx" in kwargs:
            onnx = kwargs["onnx"]
        self.embed = model.embed
        self.decoder = model.decoder
        self.model = model
        self.feats_dim = self.embed.embedding_dim
        self.num_embeddings = self.embed.num_embeddings
        self.model_name = model_name
        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = sequence_mask(max_seq_len, flip=False)
        if isinstance(model.encoder, SANMEncoder):
            self.encoder = SANMEncoder_export(model.encoder, onnx=onnx)
        else:
            assert False, "Only support samn encode."

    def forward(self, input: torch.Tensor, text_lengths: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Compute loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len)
            hidden (torch.Tensor): Target ids. (batch, len)

        """
        x = self.embed(input)
        # mask = self._target_mask(input)
        h, _ = self.encoder(x, text_lengths)
        y = self.decoder(h)
        return y

    def get_dummy_inputs(self):
        length = 120
        text_indexes = torch.randint(0, self.embed.num_embeddings, (2, length))
        text_lengths = torch.tensor([length-20, length], dtype=torch.int32)
        return (text_indexes, text_lengths)

    def get_dummy_inputs_txt(self, txt_file: str = "/mnt/workspace/data_fbank/0207/12345.wav.fea.txt"):
        import numpy as np
        fbank = np.loadtxt(txt_file)
        fbank_lengths = np.array([fbank.shape[0], ], dtype=np.int32)
        speech = torch.from_numpy(fbank[None, :, :].astype(np.float32))
        speech_lengths = torch.from_numpy(fbank_lengths.astype(np.int32))
        return (speech, speech_lengths)

    def get_input_names(self):
        return ['input', 'text_lengths']

    def get_output_names(self):
        return ['logits']

    def get_dynamic_axes(self):
        return {
            'input': {
                0: 'batch_size',
                1: 'feats_length'
            },
            'text_lengths': {
                0: 'batch_size',
            },
            'logits': {
                0: 'batch_size',
                1: 'logits_length'
            },
        }

