# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoice2Model
from cosyvoice.utils.file_utils import logging


class CosyVoice2:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CosyVoiceFrontEnd(configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir))
        self.sample_rate = configs['sample_rate']
        self.model = CosyVoice2Model(configs['flow'], configs['hift'])
        self.model.load('{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        del configs

    def tokens_to_speech(self, predicted_token, prompt_token, prompt_feat, embedding):
        wav = self.model.token2wav(token=predicted_token, prompt_token=prompt_token,
                                   prompt_feat=prompt_feat, embedding=embedding)
        return wav.cpu().detach()

    def apply_frontend(self, tts_text, prompt_text, prompt_speech_16k):
        frontend_output = self.frontend.frontend_tts(tts_text, prompt_text, prompt_speech_16k, self.sample_rate)
        prompt_token = frontend_output['prompt_speech_token']
        prompt_feat = frontend_output['prompt_speech_feat']
        embedding = frontend_output['embedding']
        return prompt_token, prompt_feat, embedding
