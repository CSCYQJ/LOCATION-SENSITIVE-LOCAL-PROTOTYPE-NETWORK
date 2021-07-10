"""Training Script"""
import os
import shutil

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from models.fewshot_grid import FewShotSeg
from dataloaders.customized import visceral_fewshot
from dataloaders.transforms import Resize, ToTensorNormalize,RandomAffine,RandomBrightness,RandomContrast,RandomGamma
from eval import load_vol_and_mask,eval
from util.utils import set_seed, CLASS_LABELS
import tqdm
from config import ex


def cal_dice(seg1,seg2):
    return (2*np.sum(seg1*seg2)+0.001)/(np.sum(seg1+seg2)+0.001)

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/val_logs', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')
    model = FewShotSeg(device=torch.device("cuda:{}".format(_config['gpu_id'])),n_grid=_config['model']['n_grid'],overlap=_config['model']['overlap'],overlap_out=_config['model']['vote'])
    model=model.to(torch.device("cuda:{}".format(_config['gpu_id'])))
    model.train()


    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'Visceral':
        make_data = visceral_fewshot
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name][_config['label_sets']]
    val_labels=CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][_config['label_sets']]
    transforms = Compose([Resize(size=_config['input_size']),
                          #RandomAffine(),
                          RandomBrightness(),
                          RandomContrast(),
                          RandomGamma()])
    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=_config['n_steps'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries']
    )
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    support_volumes=['10000132_1_CTce_ThAb.nii.gz']
    query_volumes=['10000100_1_CTce_ThAb.nii.gz',
    '10000104_1_CTce_ThAb.nii.gz',
    '10000105_1_CTce_ThAb.nii.gz',
    '10000106_1_CTce_ThAb.nii.gz',
    '10000108_1_CTce_ThAb.nii.gz',
    '10000109_1_CTce_ThAb.nii.gz',
    '10000110_1_CTce_ThAb.nii.gz',
    '10000111_1_CTce_ThAb.nii.gz',
    '10000112_1_CTce_ThAb.nii.gz',
    '10000113_1_CTce_ThAb.nii.gz',
    '10000127_1_CTce_ThAb.nii.gz',
    '10000128_1_CTce_ThAb.nii.gz',
    '10000129_1_CTce_ThAb.nii.gz',
    '10000130_1_CTce_ThAb.nii.gz',
    '10000131_1_CTce_ThAb.nii.gz',
    '10000133_1_CTce_ThAb.nii.gz',
    '10000134_1_CTce_ThAb.nii.gz',
    '10000135_1_CTce_ThAb.nii.gz',
    '10000136_1_CTce_ThAb.nii.gz']
    
    volumes_path="/ssd/qinji/PANet_Visceral/eval_dataset/volumes/"
    segs_path="/ssd/qinji/PANet_Visceral/eval_dataset/segmentations/"

    support_vol_dict,support_mask_dict=load_vol_and_mask(support_volumes,volumes_path,segs_path,labels=list(val_labels))
    query_vol_dict,query_mask_dict=load_vol_and_mask(query_volumes,volumes_path,segs_path,labels=list(val_labels))
    
    print('Successfully Load eval data!')
    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    i_iter = 0
    log_loss = {'loss': 0}
    _log.info('###### Training ######')
    best_val_dice=0
    model.eval()
    eval(model,-1,support_vol_dict,support_mask_dict,query_vol_dict,query_mask_dict,list(val_labels),os.path.join(f'{_run.observers[0].dir}/val_logs','val_logs.txt'))
    model.train()
    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        support_images = [[shot.cuda() for shot in way]
                          for way in sample_batched['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]

        query_images = [query_image.cuda()
                        for query_image in sample_batched['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

        # Forward and Backward
        optimizer.zero_grad()
        query_pred = model(support_images, support_fg_mask, support_bg_mask,
                                       query_images)
        query_loss = criterion(query_pred, query_labels)
        loss=query_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        _run.log_scalar('loss', query_loss)
        log_loss['loss'] += query_loss
        


        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            print(f'step {i_iter+1}: loss: {loss}')
            with open(os.path.join(f'{_run.observers[0].dir}/val_logs','val_logs.txt'),'a') as f:
                f.write(f'step {i_iter+1}: loss: {loss}\n')

        if (i_iter + 1) % _config['val_interval'] == 0:
            _log.info('###### Validing ######')
            model.eval()
            average_labels_dice=eval(model,i_iter,support_vol_dict,support_mask_dict,query_vol_dict,query_mask_dict,list(val_labels),os.path.join(f'{_run.observers[0].dir}/val_logs','val_logs.txt'))
            model.train()
            print(f'step {i_iter+1}: val_average_dice: {average_labels_dice}')

            if average_labels_dice>best_val_dice:
                _log.info('###### Taking snapshot ######')
                best_val_dice=average_labels_dice
                torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', 'best_val.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
