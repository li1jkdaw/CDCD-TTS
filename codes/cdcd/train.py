from dit import DiT
from diffusion import Diffusion
from trainer import Trainer
from dataset import ALL_CHARS

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True, default=None)
    parser.add_argument('--dataset_names', type=str, required=True, default=None)
    parser.add_argument('--checkpoint_folder', type=str, required=True, default=None)
    parser.add_argument("--dit_dim", type=int, required=False, default=512)
    parser.add_argument("--dit_depth", type=int, required=False, default=8)
    parser.add_argument("--dit_heads", type=int, required=False, default=8)
    parser.add_argument("--dit_dim_head", type=int, required=False, default=64)
    parser.add_argument("--dit_ff_mult", type=int, required=False, default=4)
    parser.add_argument("--text_dim", type=int, required=False, default=256)
    parser.add_argument("--diffusion_context", type=int, required=False, default=1024)
    parser.add_argument("--text_conv_layers", type=int, required=False, default=4)
    parser.add_argument('--diff_cond_min', type=float, required=False, default=0.05)
    parser.add_argument('--diff_cond_max', type=float, required=False, default=0.3)
    parser.add_argument('--mask_filler_tokens', required=False, action='store_true')
    parser.add_argument('--n_epochs', type=int, required=False, default=10)
    parser.add_argument('--num_warmup_updates', type=int, required=False, default=20000)
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-4)
    parser.add_argument('--max_batch_duration_sec', type=float, required=False, default=600)
    parser.add_argument('--max_batch_samples', type=int, required=False, default=128)
    parser.add_argument('--save_curr_every_updates', type=int, required=False, default=20000)
    parser.add_argument('--save_last_every_updates', type=int, required=False, default=2000)
    parser.add_argument('--log_every_updates', type=int, required=False, default=500)
    parser.add_argument('--keep_last_n_checkpoints', type=int, required=False, default=10)
    parser.add_argument('--random_seed', type=int, required=False, default=23)

    args = parser.parse_args()

    vocab_size = len(ALL_CHARS)
    prediction_net = DiT(dim=args.dit_dim, depth=args.dit_depth, heads=args.dit_heads, 
                         dim_head=args.dit_dim_head, ff_mult=args.dit_ff_mult, 
                         text_dim=args.text_dim, conv_layers=args.text_conv_layers,
                         mask_filler_tokens=args.mask_filler_tokens, 
                         diffusion_context=args.diffusion_context, 
                         vocab_size=vocab_size)
    diffusion = Diffusion(prediction_net, cond_min=args.diff_cond_min, cond_max=args.diff_cond_max)
    diffusion = diffusion.cuda()

    trainer = Trainer(diffusion=diffusion,
                      data_folder=args.data_folder,
                      dataset_names=args.dataset_names,
                      checkpoint_folder=args.checkpoint_folder,
                      n_epochs=args.n_epochs,
                      num_warmup_updates=args.num_warmup_updates,
                      learning_rate=args.learning_rate,
                      max_batch_duration_sec=args.max_batch_duration_sec,
                      max_batch_samples=args.max_batch_samples, 
                      save_curr_every_updates=args.save_curr_every_updates,
                      save_last_every_updates=args.save_last_every_updates,
                      log_every_updates=args.log_every_updates,
                      keep_last_n_checkpoints=args.keep_last_n_checkpoints,
                      random_seed=args.random_seed)

    trainer.train()


if __name__ == '__main__':
    main()
