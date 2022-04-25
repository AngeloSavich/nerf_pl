from sklearnex import patch_sklearn

patch_sklearn()

import os
from pytorch_lightning.accelerators import accelerator
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor, \
    StochasticWeightAveraging, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.loss = loss_dict['color'](coef=1)

        self.embedding_xyz = Embedding(hparams.N_emb_xyz)
        self.embedding_dir = Embedding(hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        self.nerf_coarse = NeRF(in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                                in_channels_dir=6 * hparams.N_emb_dir + 3)
        self.models = {'coarse': self.nerf_coarse}
        load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                                  in_channels_dir=6 * hparams.N_emb_dir + 3)
            self.models['fine'] = self.nerf_fine
            load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i + self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk,  # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'], batch['rgbs']
        results = self(rays)
        loss = self.loss(results, rgbs)

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'], batch['rgbs']
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        results = self(rays)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                              stack, self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)


def set_lr(trainer, model):
    # Determine configuration
    auto_lr = model.hparams.lr is None
    auto_batch_size = model.hparams.batch_size is None
    print(f'''
    
    Manual LR:          {model.hparams.lr}
    Manual Batch_Size:  {model.hparams.batch_size}  # if LR is none, this value is overridden
    
    ''')

    if auto_batch_size:  # Batch Size

        # Find new and override previous batch size
        # TODO: Make batch size and lr set independently instead of just based on if lr is set
        new_batch_size = trainer.tuner.scale_batch_size(
            model,
            max_trials=50,
            mode='power',
            steps_per_trial=5,
            init_val=256
        )
        model.hparams.batch_size = new_batch_size

    if auto_lr:  # Learning Rate
        # trainer.tune(model)
        model.hparams.lr = 0.0005

        # Find new and override previous learning rate
        # TODO: Does this learning rate affect the batch size determination at all and should
        #  this be run AND set first before batch size runs for more stable results? And then vice versa, does batch
        #  size have an impact on learning rate?
        lr_finder = trainer.tuner.lr_find(
            model,
            num_training=model.hparams.lr_num_tests,
            min_lr=1e-8,
            max_lr=1,
        )
        model.hparams.lr = lr_finder.suggestion()

        # Display learning rate results graph
        fig = lr_finder.plot(suggest=True)  # Plot
        fig.show()
        fig.savefig('auto_lr_estimation_graph.png')

    print(f'''
    Learning Rate : {model.hparams.lr},
    Mode: {'auto' if auto_lr else 'manual'}
    ''', flush=True)

    # TODO: Make batch size and lr set independently instead of just based on if lr is set
    print(f'''
    Batch Size : {model.hparams.batch_size},
    Mode: {'auto' if auto_batch_size else 'manual'}
    ''', flush=True)


# TODO: Implement Stachastic Weight Averaging?
# TODO: Make it save checkpoints in the data repo
# TODO: Auto batch size
def main(hparams):
    # TODO - Move checkpoint generation to it's own file / place / namespace
    cb_ckpt_top = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}/top5/',
                                  filename='top5-{epoch:0>3d}-{step:d}',
                                  every_n_epochs=1,
                                  save_top_k=6,
                                  monitor='val/psnr',
                                  mode='max',
                                  save_on_train_epoch_end=True)

    cb_every_epoch = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}/all_epochs',
                                     filename='all-{epoch:0>3d}-{step:d}',
                                     every_n_epochs=1,
                                     save_top_k=-1,
                                     save_on_train_epoch_end=False)

    cb_every_epoch_end = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}/all_epochs',
                                         filename='all-{epoch:0>3d}-{step:d}-end',
                                         every_n_epochs=1,
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True)

    # cb_ckpt_min_loss_train = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}/',
    #                                          filename='top_min_loss_train-{epoch:d}-{step:d}',
    #                                          every_n_epochs=1,
    #                                          save_top_k=1,
    #                                          monitor='train/loss',
    #                                          mode='min',
    #                                          save_on_train_epoch_end = True)

    cb_ckpt_min_loss_mean = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}/',
                                            filename='top_min_loss_mean-{epoch:d}-{step:d}',
                                            every_n_epochs=1,
                                            save_top_k=1,
                                            monitor='val/loss',
                                            mode='min',
                                            save_on_train_epoch_end=True)

    # cb_ckpt_max_psnr_train = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}/',
    #                                          filename='top_max_psnr_train-{epoch:d}-{step:d}',
    #                                          every_n_epochs=1,
    #                                          save_top_k=1,
    #                                          monitor='train/psnr',
    #                                          mode='max',
    #                                          save_on_train_epoch_end = True)

    cb_ckpt_max_psnr_mean = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}/',
                                            filename='top_max_psnr_mean-{epoch:d}',
                                            every_n_epochs=1,
                                            save_top_k=1,
                                            monitor='val/psnr',
                                            mode='max',
                                            save_on_train_epoch_end=True)

    cb_ckpt_latest = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}/',
                                     filename='last-{epoch:d}',
                                     save_last=True)


    cb_early_stop_loss = EarlyStopping(check_finite=True, monitor='train/loss')
    cb_early_stop_psnr = EarlyStopping(check_finite=True, monitor='train/psnr')

    pbar = TQDMProgressBar(refresh_rate=1)

    lr_bar = LearningRateMonitor(logging_interval='step',
                                 log_momentum=True)

    callbacks = [
        # StochasticWeightAveraging(swa_lrs=1e-2),
        cb_ckpt_top, cb_ckpt_latest,
        cb_every_epoch, cb_every_epoch_end,
        cb_ckpt_min_loss_mean, cb_ckpt_max_psnr_mean,
        # cb_ckpt_min_loss_train, cb_ckpt_max_psnr_train,
        cb_early_stop_loss, cb_early_stop_psnr,
        pbar, lr_bar,
    ]

    #################################################
    system = NeRFSystem(hparams)

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    # trainer_params = {}
    # if hparams.mixed_precision is not None:
    #     trainer_params['precision'] = 16

    trainer = Trainer(
        # auto_lr_find=True,  # TODO: Does this param still need to be active? Also... move it lower to the bottom.
        # auto_scale_batch_size='binsearch',
        precision=16,
        max_epochs=hparams.num_epochs,
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=False,
        accelerator='auto',
        devices=hparams.num_gpus,
        num_sanity_val_steps=1,
        benchmark=True,
        profiler='simple' if hparams.num_gpus == 1 else None,
        strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus > 1 else None,
        # **trainer_params
    )

    # Auto Find Learning Rate: tune trainer
    set_lr(trainer, system)

    # trainer.tune(
    #     system
    # )  # Tunes for batch size, but may also retune for lr # TODO: Check if you want to move this before set_lr

    fit_params = {}
    if hparams.ckpt_path is not None:
        fit_params['ckpt_path'] = hparams.ckpt_path

    trainer.fit(system, **fit_params)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
