import os
import math
import numpy as np
import gc

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, SequentialSampler

from diffusion import Diffusion
from dataset import CustomDataset, CustomCollate, CustomBatchSampler


class Trainer:
    def __init__(self,
                 diffusion: Diffusion,
                 data_folder: str,
                 dataset_names: str,
                 checkpoint_folder: str,
                 n_epochs: int = 10,
                 num_warmup_updates: int = 20000,
                 learning_rate: float = 1e-4,
                 max_batch_duration_sec: float = 600,
                 max_batch_samples: int = 128,
                 num_workers: int = 4,
                 grad_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 save_curr_every_updates: int = 20000,
                 save_last_every_updates: int = 2000,
                 log_every_updates: int = 500,
                 log_smooth_factor: float = 0.999,
                 keep_last_n_checkpoints: int = 10,
                 random_seed: int = 23,
                 accelerate_kwargs: dict = dict()):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],
                                       gradient_accumulation_steps=grad_accumulation_steps,
                                       **accelerate_kwargs)

        self.diffusion = diffusion
        if self.is_main:
            num_params = 0
            for name, param in self.diffusion.named_parameters():
                if param.requires_grad:
                    num_params += np.prod(param.detach().cpu().numpy().shape)
            print("Number of DPM parameters is %.1fm" % (num_params / 1000000))

        dataset = CustomDataset(data_folder, dataset_names, verbose=self.is_main)
        if self.is_main:
            print("Overall duration is %.1fk hours.\n" % (dataset.duration / 3600000))

        collate_fn = CustomCollate()
        sampler = SequentialSampler(dataset)
        batch_sampler = CustomBatchSampler(sampler, max_duration=max_batch_duration_sec, 
                                           max_samples=max_batch_samples, random_seed=random_seed, 
                                           drop_residual=True, verbose=self.is_main)
        self.dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_sampler=batch_sampler, 
                                     num_workers=num_workers, pin_memory=True, persistent_workers=True)
        self.accelerator.even_batches = False

        self.optimizer = AdamW(self.diffusion.parameters(), lr=learning_rate)
        warmup_steps = num_warmup_updates * self.accelerator.num_processes
        total_steps = math.ceil(len(self.dataloader) / grad_accumulation_steps) * n_epochs
        decay_steps = total_steps - warmup_steps
        num_decay_updates = math.ceil(total_steps / self.accelerator.num_processes) - num_warmup_updates
        if self.is_main:
            print("Number of warmup updates: %.1fk" % (num_warmup_updates / 1000))
            print("Number of decay updates: %.1fk\n" % (num_decay_updates / 1000))
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, 
                                    total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, 
                                   total_iters=decay_steps)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], 
                                      milestones=[warmup_steps])

        self.diffusion, self.dataloader = self.accelerator.prepare(self.diffusion, self.dataloader)
        self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)

        self.n_epochs = n_epochs
        self.num_warmup_updates = num_warmup_updates
        self.learning_rate = learning_rate
        self.max_batch_duration_sec = max_batch_duration_sec
        self.max_batch_samples = max_batch_samples
        self.num_workers = num_workers
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_curr_every_updates = save_curr_every_updates
        self.save_last_every_updates = save_last_every_updates
        self.log_every_updates = log_every_updates
        self.log_smooth_factor = log_smooth_factor
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.checkpoint_folder = checkpoint_folder
        self.random_seed = random_seed

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    @property
    def num_processes(self):
        return self.accelerator.num_processes

    def save_checkpoint(self, update, is_last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(model_state_dict=self.accelerator.unwrap_model(self.diffusion).state_dict(),
                              optimizer_state_dict=self.optimizer.state_dict(),
                              scheduler_state_dict=self.scheduler.state_dict(),
                              update=update)
            if not os.path.exists(self.checkpoint_folder):
                os.makedirs(self.checkpoint_folder)
            if is_last:
                self.accelerator.save(checkpoint, "%s/model_last.pt" % self.checkpoint_folder)
                print("Last checkpoint at update %d saved." % update)
            else:
                self.accelerator.save(checkpoint, "%s/model_%d.pt" % (self.checkpoint_folder, update))
                checkpoints = [fname for fname in os.listdir(self.checkpoint_folder)
                               if fname.startswith("model_") and fname.endswith(".pt") 
                               and fname != "model_last.pt" and fname != "model_pretrained.pt"]
                checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                while len(checkpoints) > self.keep_last_n_checkpoints:
                    oldest_checkpoint = checkpoints.pop(0)
                    os.remove(os.path.join(self.checkpoint_folder, oldest_checkpoint))
                    print("Old checkpoint %s removed." % oldest_checkpoint)

    def load_checkpoint(self):
        if (self.checkpoint_folder is None) or (not os.path.exists(self.checkpoint_folder)):
            return 0

        if not any(filename.endswith(".pt") for filename in os.listdir(self.checkpoint_folder)):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_folder):
            latest_checkpoint = "model_last.pt"
        else:
            checkpoints = [fname for fname in os.listdir(self.checkpoint_folder)
                           if fname.startswith("model_") and fname.endswith(".pt")
                           and fname != "model_last.pt" and fname != "model_pretrained.pt"]
            if len(checkpoints) == 0:
                return 0
            checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            latest_checkpoint = checkpoints[-1]

        checkpoint = torch.load("%s/%s" % (self.checkpoint_folder, latest_checkpoint), 
                                weights_only=True, map_location="cpu")

        self.accelerator.unwrap_model(self.diffusion).load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        update = checkpoint["update"]

        del checkpoint
        gc.collect()
        return update

    def train(self):
        self.diffusion.train()

        if self.is_main:
            log_str = "Training started."
            print(log_str)
            with open(os.path.join(self.checkpoint_folder, "log.txt"), "a") as f:
                f.write(log_str + "\n")

        start_update = self.load_checkpoint()

        if self.num_processes == 1:
            if not self.diffusion.exists_ema_model():
                self.diffusion.init_ema_model()
        else:
            if not self.diffusion.module.exists_ema_model():
                self.diffusion.module.init_ema_model()

        global_update = start_update
        orig_epoch_step = len(self.dataloader)
        start_step = start_update * self.grad_accumulation_steps
        skipped_epoch = int(start_step // orig_epoch_step)
        skipped_batch = start_step % orig_epoch_step
        skipped_dataloader = self.accelerator.skip_first_batches(self.dataloader, num_batches=skipped_batch)

        loss_smooth = None
        for epoch in range(skipped_epoch, self.n_epochs):
            if self.is_main:
                log_str = "Epoch %d started." % (epoch + 1)
                print(log_str)
                with open(os.path.join(self.checkpoint_folder, "log.txt"), "a") as f:
                    f.write(log_str + "\n")

            if epoch == skipped_epoch:
                current_dataloader = skipped_dataloader
            else:
                current_dataloader = self.dataloader

            if self.num_processes == 1:
                self.dataloader.batch_sampler.set_epoch(epoch)
            else:
                self.dataloader.batch_sampler.batch_sampler.set_epoch(epoch)

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.diffusion):
                    text, tokens, vectors = batch["text"], batch["tokens"], batch["vectors"]
                    lengths = batch["lengths"]

                    if self.num_processes == 1:
                        loss = self.diffusion.compute_diff_loss(vectors, text, lengths, tokens)
                    else:
                        loss = self.diffusion.module.compute_diff_loss(vectors, text, lengths, tokens)

                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.diffusion.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    global_update += 1
                    if self.num_processes == 1:
                        self.diffusion.update_ema_model()
                    else:
                        self.diffusion.module.update_ema_model()

                if self.is_main and self.accelerator.sync_gradients:
                    if loss_smooth is None:
                        loss_smooth = loss.item()
                    else:
                        loss_smooth = self.log_smooth_factor*loss_smooth
                        loss_smooth += (1 - self.log_smooth_factor)*loss.item()

                    if global_update % self.log_every_updates == 0:         
                        lr = self.scheduler.get_last_lr()[0] 
                        log_str = "Update %d [lr %.8f]: " % (global_update, lr)
                        log_str += "diffusion loss = %.4f" % loss_smooth
                        print(log_str)
                        with open(os.path.join(self.checkpoint_folder, "log.txt"), "a") as f:
                            f.write(log_str + "\n")

                if global_update % self.save_last_every_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, is_last=True)
                    torch.cuda.empty_cache()

                if global_update % self.save_curr_every_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

        self.save_checkpoint(global_update, is_last=True)

        if self.is_main:
            log_str = "Training finished!"
            print(log_str)
            with open(os.path.join(self.checkpoint_folder, "log.txt"), "a") as f:
                f.write(log_str)

        self.accelerator.end_training()
