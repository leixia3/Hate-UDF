import argparse
import datetime
import os
import torch
import wandb
from datasets import CustomCollator, load_dataset
from engine import create_model

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

def str2bool(v):
    """
    src: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse 
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arg_parser():

    parser = argparse.ArgumentParser(description='Traning and evaluation script for hateful meme classification')

    # dataset parameters
    parser.add_argument('--dataset', default='original', choices=['original', 'masked', 'inpainted', 'tamil', 'prop'])
    parser.add_argument('--labels', default='original', choices=['original', 'fine_grained', 'fine_grained_gold'])
    parser.add_argument('--image_size', type=int, default=224)

    # model parameters
    parser.add_argument('--multilingual_tokenizer_path', type=str, default='none', choices=['none', 'bert-base-multilingual-uncased', 'xlm-roberta-base'])
    parser.add_argument('--clip_pretrained_model', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--local_pretrained_weights', type=str, default='none')
    parser.add_argument('--caption_mode', type=str, default='none', choices=['none', 'replace_image', 'replace_text', 'concat_with_text', 'parallel_mean', 'parallel_max', 'parallel_align'])
    parser.add_argument('--use_pretrained_map', default=False, type=str2bool)
    parser.add_argument('--num_mapping_layers', default=1, type=int)
    parser.add_argument('--map_dim', default=768, type=int)
    parser.add_argument('--fusion', default='clip', choices=['align', 'align_shuffle', 'concat', 'cross', 'cross_nd', 'align_concat', 'weighted_align', 'weighted_align_shuffle', 'weighted_concat', 'weighted_cross', 'weighted_cross_nd', 'attention_m'])
    parser.add_argument('--num_pre_output_layers', default=1, type=int)
    parser.add_argument('--drop_probs', type=float, nargs=3, default=[0.1, 0.4, 0.2], help="Set drop probabilities for map, fusion, pre_output")
    parser.add_argument('--image_encoder', type=str, default='clip')
    parser.add_argument('--text_encoder', type=str, default='clip')
    parser.add_argument('--freeze_image_encoder', type=str2bool, default=True)
    parser.add_argument('--freeze_text_encoder', type=str2bool, default=True)
    parser.add_argument('--from_checkpoint', type=str, default='none')
    parser.add_argument('--with_pro_cap', action='store_true', default=False, help='Using Pro-Cap')
    parser.add_argument('--without_val', action='store_true', default=False, help='Without validation')
    parser.add_argument('--test_only', action='store_true', default=False, help='Test only and point out the checkpoint filename')
    # training parameters
    parser.add_argument('--remove_matches', type=str2bool, default=False)
    parser.add_argument('--gpus', default='0', help='GPU ids concatenated with space')
    parser.add_argument('--strategy', default='auto')
    parser.add_argument('--limit_train_batches', default=1.0)
    parser.add_argument('--limit_val_batches', default=1.0)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--max_epochs', type=int, default=-1)
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--val_check_interval', default=1.0)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_image_loss', type=float, default=1.0)
    parser.add_argument('--weight_text_loss', type=float, default=1.0)
    parser.add_argument('--weight_fine_grained_loss', type=float, default=1.0)
    parser.add_argument('--weight_super_loss', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_clip_val', type=float, default=0.1)

    # other parameters
    # parser.add_argument('--eval_split', default='test_seen', choices=['test_seen', 'val_seen'])

    return parser

def main(args):
    # TODO: 由于开启fine-grained后需要验证集，需要设计--labels与--without_val的关系
    # 通过设置--labels设置是否开启fine-grained
    # load dataset
    if args.dataset in ['original', 'masked', 'inpainted']:
        dataset_train = load_dataset(args=args, split='train')
        if not args.without_val:
            dataset_val = load_dataset(args=args, split='test_seen')
            dataset_val_unseen = load_dataset(args=args, split='dev_unseen')
        dataset_test = load_dataset(args=args, split='test_seen')
        dataset_test_unseen = load_dataset(args=args, split='test_unseen')
    elif args.dataset == 'tamil':
        dataset_train = load_dataset(args=args, split='train')
        dataset_val = load_dataset(args=args, split='test')
    elif args.dataset == 'prop':
        dataset_train = load_dataset(args=args, split='train')
        dataset_val = load_dataset(args=args, split='val')
        dataset_test = load_dataset(args=args, split='test')
    print("Number of training examples:", len(dataset_train))
    if not args.without_val:
        print("Number of validation examples:", len(dataset_val))
    if args.dataset in ['original', 'masked', 'inpainted']:
        print("Number of test examples:", len(dataset_test))
        if not args.without_val:
            print("Number of validation examples (unseen):", len(dataset_val_unseen))
        print("Number of test examples (unseen):", len(dataset_test_unseen))
    elif args.dataset == 'prop':
        print("Number of test examples:", len(dataset_test))
    print("Sample item:", dataset_train[0])
    print("Image size:", dataset_train[0]['image'].size)

    # load dataloader
    num_cpus = min(args.batch_size, 16) #(multiprocessing.cpu_count() // len(args.gpus))-1
    if args.dataset == 'tamil' and args.caption_mode != 'none':
        multilingual_tokenizer_path = args.multilingual_tokenizer_path
    else:
        multilingual_tokenizer_path = 'none'
    collator = CustomCollator(args, dataset_train.fine_grained_labels, multilingual_tokenizer_path=multilingual_tokenizer_path)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_cpus, collate_fn=collator, drop_last=True)
    if not args.without_val:
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator, drop_last=True)
    if args.dataset in ['original', 'masked', 'inpainted']:
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator, drop_last=True)
        if not args.without_val:
            dataloader_val_unseen = DataLoader(dataset_val_unseen, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator, drop_last=True)
        dataloader_test_unseen = DataLoader(dataset_test_unseen, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator, drop_last=True)
    elif args.dataset == 'prop':
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator, drop_last=True)
    
    # create model
    seed_everything(42, workers=True)
    model = create_model(args, dataset_train.fine_grained_labels)

    # sanity check
    # batch = next(iter(dataloader_train))
    # output = model(batch)
    # print(output)

    if args.dataset == 'prop':
        monitor="val/f1"
        project="meme-prop-v2"
    elif args.dataset == 'tamil':
        monitor="val/f1"
        project="meme-tamil-v2"
    else:
        if not args.without_val:
            monitor="val/f1"
        else:
            monitor="train/auroc"
        project="meme-v2"

    wandb_logger = WandbLogger(project=project, config=args)
    num_params = {f'param_{n}':p.numel() for n, p in model.named_parameters() if p.requires_grad}
    wandb_logger.experiment.config.update(num_params)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', filename='-{epoch:02d}',  monitor=monitor, mode='max', verbose=True, save_weights_only=True, save_top_k=1, save_last=False)

    trainer = Trainer(max_epochs=args.max_epochs, max_steps=args.max_steps,
                      gradient_clip_val=args.gradient_clip_val,
                      logger=wandb_logger, log_every_n_steps=args.log_every_n_steps,
                      val_check_interval=args.val_check_interval,
                      strategy=args.strategy, callbacks=[checkpoint_callback],
                      limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
                      deterministic=True, gpus=args.gpus)

    model.compute_fine_grained_metrics = True
    if args.test_only and args.from_checkpoint != 'none':
        if args.dataset in ['original', 'masked', 'inpainted']:
            if not args.without_val:
                trainer.test(model, dataloaders=[dataloader_test, dataloader_val])
            else:
                trainer.test(model, dataloaders=[dataloader_test])
        elif args.dataset == 'tamil':
            trainer.test(model, dataloaders=[dataloader_val, dataloader_val])
        elif args.dataset == 'prop':
            trainer.test(model, dataloaders=[dataloader_test, dataloader_val])

    else:
        if not args.without_val:
            trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
        else:
            trainer.fit(model, train_dataloaders=dataloader_train)
        if args.dataset in ['original', 'masked', 'inpainted']:
            if not args.without_val:
                trainer.test(ckpt_path='best', dataloaders=[dataloader_test, dataloader_val])
            else:
                trainer.test(ckpt_path='best', dataloaders=[dataloader_test])
        elif args.dataset == 'tamil':
            trainer.test(ckpt_path='best', dataloaders=[dataloader_val, dataloader_val])
        elif args.dataset == 'prop':
            trainer.test(ckpt_path='best', dataloaders=[dataloader_test, dataloader_val])


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    args.gpus = [int(id_) for id_ in args.gpus.split()]
    if args.strategy == 'ddp':
        # args.strategy = DDPPlugin(find_unused_parameters=False)
        args.strategy = DDPStrategy(find_unused_parameters=False)
    elif args.strategy == 'none':
        args.strategy = None

    if args.multilingual_tokenizer_path != 'none':
        if args.text_encoder == 'clip':
            args.text_encoder = args.multilingual_tokenizer_path

    main(args)
