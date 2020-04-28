import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import auto_fp16, force_fp32, mask_target
from mmdet.ops.carafe import CARAFEPack
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, build_upsample_layer
import pickle
import time
import pdb

@HEADS.register_module
class MGANHead(nn.Module):

    def __init__(self,
                 num_convs=2,
                 roi_feat_size=7,
                 in_channels=512,
                 conv_out_channels=512,
                 conv_cfg=None,
                 norm_cfg=None,
                 class_agnostic=False,
                 loss_mask=dict(
                     type='CrossEntropyLoss', # todo: binary?
                     use_mask=True,
                     loss_weight=0.3), 
                 loss_occ=dict(
                     type='CrossEntropyLoss',
                     use_mask=True,
                     loss_weight=1.0
                 )):
        super(MGANHead, self).__init__()
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        logits_in_channel = self.conv_out_channels
        self.conv_logits = nn.Conv2d(logits_in_channel, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        self.class_agnostic = class_agnostic
        self.loss_mask = build_loss(loss_mask)
        self.loss_occ = build_loss(loss_occ)

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        # x = self.conv_logits(x).sigmoid() * x
        x_fpm = self.conv_logits(x).sigmoid()
        x = x_fpm * x

        # with open('/mmdetection/result/attent_vis/attention_fpm/' + str(time.time()), 'wb') as pkl_file:
        #     pickle.dump(x_fpm, pkl_file)
        return x

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def get_bbox_target(self, sampling_results, gt_bboxes, gt_labels,
            rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        # neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1
        cls_reg_targets = bbox_target(
                pos_proposals,
                # neg_proposals,
                pos_gt_bboxes,
                pos_gt_labels,
                rcnn_train_cfg,
                reg_classes,
                target_means=self.target_means,
                target_stds=self.target_stds)
        return cls_reg_targets

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    def loss_occ(self,
             cls_score,
             labels,
             occ_weights,
             ):
        loss = dict()
        loss_sum = []
        batch_size = cls_score.shape[0]
        for i in range(batch_size):
            loss_i = F.cross_entropy(cls_score[i:i+1], labels[i:i+1])
            occ = occ_weights[i]
            occ_level = 1 - sum(sum(occ)) / (occ.shape[0] * occ.shape[1])
            loss_sum.append(occ_level * loss_i)

        loss["occ"] = sum(loss_sum) / batch_size
        
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)
        # when enabling mixed precision training, mask_pred may be float16
        # numpy array
        mask_pred = mask_pred.astype(np.float32)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            if not isinstance(scale_factor, (float, np.ndarray)):
                scale_factor = scale_factor.cpu().numpy()
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)

            if rcnn_test_cfg.get('crop_mask', False):
                im_mask = bbox_mask
            else:
                im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask

            if rcnn_test_cfg.get('rle_mask_encode', True):
                rle = mask_util.encode(
                    np.array(im_mask[:, :, np.newaxis], order='F'))[0]
                cls_segms[label - 1].append(rle)
            else:
                cls_segms[label - 1].append(im_mask)

        return cls_segms

