from collections import OrderedDict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

#from .vgg import Encoder
from .resnet import ResNet,Res_Deeplab,load_resnet50_param

def gen_indices(i, k, s):
    assert i >= k, 'Sample size has to be bigger than the patch size'
    for j in range(0, i - k + 1, s):
        yield j
    if j + k < i:
        yield i - k

def get_grid(data_shape,grid_num,overlap=False):
    grids = []
    i_y, i_x = data_shape
    k_y, k_x = i_y//grid_num,i_x//grid_num
    if overlap==False:
        s_y,s_x=k_y, k_x
    else:
        s_y,s_x=k_y//2, k_x//2
    y_steps = gen_indices(i_y, k_y, s_y)
    for y in y_steps:
        x_steps = gen_indices(i_x, k_x, s_x)
        for x in x_steps:
            grid_idx = (
                    slice(y, y + k_y),
                    slice(x, x + k_x)
            )
            grids.append(grid_idx)
    return grids

class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, in_channels=3, device=torch.device("cuda:0"),n_grid=4,overlap=True,overlap_out='average'):
        super().__init__()
        self.device=device
        self.grid_num=n_grid
        self.overlap=overlap
        self.overlap_out=overlap_out
        # Encoder
        
        self.encoder=Res_Deeplab()
        self.encoder=load_resnet50_param(self.encoder)

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)
        
        
                                
        img_fts = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]
        #print(img_fts.shape)
        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        back_mask = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'
        grids=get_grid(fore_mask.shape[-2:],grid_num=self.grid_num,overlap=self.overlap)
        ###### Compute loss ######
        outputs = []
        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_grid_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]],grids)
                            for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_grid_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             back_mask[way, shot, [epi]],grids)
                            for shot in range(n_shots)] for way in range(n_ways)]
            ###### Obtain the prototypes######
            fg_grid_prototypes, bg_grid_prototype = self.getPrototype(supp_fg_grid_fts, supp_bg_grid_fts)
            ###### Compute the distance ######
            prototypes = [bg_grid_prototype,] + fg_grid_prototypes
            upsample_qry_fts=F.interpolate(qry_fts[:, epi],size=img_size,mode='bilinear')
            dist = [self.calDist(upsample_qry_fts, prototype,grids,img_size,overlap_out=self.overlap_out) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H x W
            outputs.append(pred)
        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        return output

    def calDist(self,fts,prototype,grids,img_size,overlap_out='average',scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        H,W=img_size
        dist= torch.zeros((fts.shape[0],H,W)).float().to(self.device)
        if overlap_out=='average':
            weights=torch.zeros((fts.shape[0],H,W)).float().to(self.device)
            for (i,grid) in enumerate(grids):
                weights[:,grid[0],grid[1]]=weights[:,grid[0],grid[1]]+torch.ones_like(weights[:,grid[0],grid[1]]).float().to(self.device)
                dist[:,grid[0],grid[1]] = dist[:,grid[0],grid[1]]+F.cosine_similarity(fts[:,:,grid[0],grid[1]], prototype[i][..., None, None], dim=1) * scaler
            dist=dist/weights
        elif overlap_out=='max':
            dist= dist-1
            for (i,grid) in enumerate(grids):
                new_local_dist=F.cosine_similarity(fts[:,:,grid[0],grid[1]], prototype[i][..., None, None], dim=1) * scaler
                dist[:,grid[0],grid[1]] = (dist[:,grid[0],grid[1]]>new_local_dist)*dist[:,grid[0],grid[1]]+(dist[:,grid[0],grid[1]]<=new_local_dist)*new_local_dist
        elif overlap_out=='cover':
            for (i,grid) in enumerate(grids):
                dist[:,grid[0],grid[1]]=F.cosine_similarity(fts[:,:,grid[0],grid[1]], prototype[i][..., None, None], dim=1) * scaler
        return dist



    def getFeatures(self, fts, mask, grids):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        masked_grid_fts=[]
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts=fts * mask[None, ...]
        for grid in grids:
            if torch.sum(mask[:,grid[0],grid[1]])<=10:
                masked_grid_fts.append(torch.zeros((1,fts.shape[1])).float().to(self.device))
            else:
                masked_grid_fts.append(torch.sum(masked_fts[:,:,grid[0],grid[1]], dim=(2, 3)) \
            / (mask[None,:,grid[0],grid[1]].sum(dim=(2, 3)) + 1e-5)) # 1 x C
        return masked_grid_fts

    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [grids_num x 1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [grids_num x 1 x C]
        """
        n_ways, n_shots , n_grids = len(fg_fts), len(fg_fts[0]),len(fg_fts[0][0])
        C=fg_fts[0][0][0].shape[-1]
        fg_grid_prototypes=[[torch.zeros((1,C)).float().to(self.device) for grid in range(n_grids)] for way in range(n_ways)]
        bg_grid_prototype=[torch.zeros((1,C)).float().to(self.device) for grid in range(n_grids)]
        for way in range(n_ways):
            for i in range(n_grids):
                for j in range(n_shots):
                    fg_grid_prototypes[way][i]+=fg_fts[way][j][i]
                    bg_grid_prototype[i]+=bg_fts[way][j][i]
                fg_grid_prototypes[way][i]/=n_shots
        for i in range(n_grids):
            bg_grid_prototype[i]/=(n_shots*n_ways)
        return fg_grid_prototypes, bg_grid_prototype

'''if __name__ == '__main__':

    model = FewShotSeg(device=torch.device("cpu"),n_grid=4,overlap_out='cover')
    support_slices = [[torch.rand((3,3,512,512))]]
    support_fg_masks = [[torch.ones((3,512,512))]]
    support_bg_masks = [[torch.ones((3,512,512))]]
    query_slices = [torch.rand((3,3,512,512))]
    pred = model(support_slices, support_fg_masks, support_bg_masks,query_slices)
    print(pred.shape)'''