import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from utils import fsq_token_to_vector


LETTERS = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, 
           "k": 10, "l": 11, "m": 12, "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, 
           "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24, "z": 25}
PUNCTUATION = {".": 26, ",": 27, "?": 28, "!": 29, ":": 30, ";": 31, "'": 32}
ALL_CHARS = {**LETTERS, **PUNCTUATION, " ": 33}

PREFILLER_TOKEN = -1


class CustomDataset(Dataset):
    def __init__(self, data_folder, dataset_names, fsq_base=3, fsq_dim=8, 
                 verbose=True):
        super().__init__()

        self.fsq_base = fsq_base
        self.fsq_dim = fsq_dim
        self.vocab_size = len(ALL_CHARS)

        self.duration = 0.0
        self.data = []

        self.data_folder = data_folder
        self.dataset_names = [dname.strip() for dname in dataset_names.split(",")]
        for dataset_id, dataset_name in enumerate(self.dataset_names):
            if verbose:
                print("processing dataset %s..." % dataset_name)
            data_path = os.path.join(data_folder, dataset_name)
            info_files = [fname for fname in os.listdir(data_path)
                          if fname.endswith("_info.txt")]
            for info_file in info_files:
                npz_name = info_file.replace("_info.txt", "")
                with open(os.path.join(data_path, info_file), "r") as f:
                    lines = f.readlines()
                lines = [line.strip() for line in lines if len(line.strip()) > 0]
                for audio_id, line in enumerate(lines):
                    duration = float(line.split("|")[0])
                    data_elem = {"dataset_id": dataset_id,
                                 "npz_name": npz_name,
                                 "audio_id": audio_id,
                                 "duration": duration}
                    self.duration += duration
                    self.data.append(data_elem)
            if verbose:
                print("cumulative duration is %.1fk hours.\n" % (self.duration / 3600000))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_elem = self.data[index]
        duration = data_elem["duration"]
        dataset_name = self.dataset_names[data_elem["dataset_id"]]
        data_path = os.path.join(self.data_folder, dataset_name)
        speech_path = os.path.join(data_path, data_elem["npz_name"] + "_speech.npz")
        text_path = os.path.join(data_path, data_elem["npz_name"] + "_text.npz")
        speech_arrays = np.load(speech_path)
        text_arrays = np.load(text_path)
        speech_tokens = speech_arrays["arr_%d" % data_elem["audio_id"]]
        text_tokens = text_arrays["arr_%d" % data_elem["audio_id"]]
        text_tokens = text_tokens[text_tokens < len(ALL_CHARS)]
        speech_vector = [fsq_token_to_vector(speech_tokens[i], base=self.fsq_base, dim=self.fsq_dim)
                         for i in range(len(speech_tokens))]
        speech_vector = torch.stack(speech_vector, dim=0).float()
        speech_tokens = torch.from_numpy(speech_tokens).long()
        text = torch.from_numpy(text_tokens).long()
        out_dict = {"speech_tokens": speech_tokens, "speech_vector": speech_vector,
                    "text": text, "duration": duration}
        return out_dict

 
class CustomCollate(object):
    def __call__(self, batch):
        max_text_len = max([item["text"].shape[0] for item in batch])
        max_speech_len = max([item["speech_tokens"].shape[0] for item in batch])
        text_list, speech_token_list, speech_vector_list = [], [], []
        speech_lengths = []
        for item in batch:
            text = item["text"]
            speech_tokens = item["speech_tokens"]
            speech_vector = item["speech_vector"]
            text_length = text.shape[0]
            speech_length = speech_tokens.shape[0]

            text_list.append(F.pad(text.unsqueeze(0), 
                                   (0, max_text_len - text_length), value=PREFILLER_TOKEN))
            speech_token_list.append(F.pad(speech_tokens.unsqueeze(0), 
                                           (0, max_speech_len - speech_length), value=0))
            speech_vector_list.append(F.pad(speech_vector.unsqueeze(0), 
                                            (0, 0, 0, max_speech_len - speech_length), value=0.0))
            speech_lengths.append(speech_length)

        text = torch.cat(text_list, dim=0)
        speech_tokens = torch.cat(speech_token_list, dim=0)
        speech_vector = torch.cat(speech_vector_list, dim=0)
        speech_lengths = torch.LongTensor(speech_lengths)
        out_dict = {"text": text, "tokens": speech_tokens, 
                    "vectors": speech_vector, "lengths": speech_lengths}
        return out_dict


class CustomBatchSampler(Sampler):
    def __init__(self, seq_sampler, max_duration, max_samples=0, random_seed=None, 
                 drop_residual=True, verbose=True):
        self.sampler = seq_sampler
        self.max_duration = max_duration
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.drop_residual = drop_residual
        self.drop_last = True   # for the reason unknown
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        if verbose:
            desc = "Sorting audio by duration..."
            idx_iterator = tqdm(self.sampler, desc=desc)
        else:
            idx_iterator = self.sampler
        for idx in idx_iterator:
            indices.append((idx, data_source.data[idx]["duration"]))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_duration = 0.0
        if verbose:
            desc = "Creating batches [max duration = %.0f secs/gpu]..." % max_duration
            batch_iterator = tqdm(indices, desc=desc)
        else:
            batch_iterator = indices
        for idx, dur in batch_iterator:
            if batch_duration + dur <= max_duration and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_duration += dur
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if dur <= max_duration:
                    batch = [idx]
                    batch_duration = dur
                else:
                    batch = []
                    batch_duration = 0.0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)
