"""
produce a csv wit the following columns for each example in the dataset:

dicom_sent_id,patient_id,study_id,dicom_id,sent_id,sentence,bbox_names,sent_labels,sent_contexts,bboxes,
auroc,avg_precision,attn_entropy,no_attn_weight,local_sims,global_sims,precision_at_0.050000,recall_at_0.050000,
f1_at_0.050000,iou_at_0.050000,precision_at_0.100000,recall_at_0.100000,f1_at_0.100000,iou_at_0.100000,
precision_at_0.200000,recall_at_0.200000,f1_at_0.200000,iou_at_0.200000,precision_at_0.300000,recall_at_0.300000,
f1_at_0.300000,iou_at_0.300000

from the data and the uniter outputs obtained from validate_wrong_lung.py:

a npz file named as <dicom_sent_id>.npz in an output folder, where each file contains the following numpy arrays:
-attention
-bboxes
-pixel_level_attention
"""
from preprocess_imagenome import process_bboxes, bbox_to_mask
import json
import os
import numpy as np
import torch
from torchmetrics.functional import roc, precision_recall_curve, auroc, average_precision, precision_recall, f1
from torch.distributions.categorical import Categorical
import pandas as pd
from tqdm import tqdm


def discrete_entropy(dist):
    # note if no-attn is not turned on, this makes no difference because 1 - dist.sum() = 0
    dist = torch.cat([get_no_attn_weight(dist).unsqueeze(0), dist], 0)
    return Categorical(dist).entropy()


def get_no_attn_weight(dist):
    return 1 - dist.sum(-1)


def sent_bboxes_to_segmentation_label(shape, sent_bboxes):
    segmentation_label = torch.zeros(shape, dtype=torch.bool)
    for bbox in sent_bboxes:
        segmentation_label = segmentation_label | bbox_to_mask(bbox, shape)
    return segmentation_label


class Metrics:
    def __init__(self, percentile_thresholds=[.05, .1, .2, .3]):
        self.attn_bbox_metrics = {
            'roc_curve': roc,
            'pr_curve': precision_recall_curve,
            'auroc': auroc,
            'avg_precision': average_precision,
        }
        self.percentile_thresholds = percentile_thresholds
        self.attn_entropy = discrete_entropy
        self.no_attn_weight = get_no_attn_weight

    def __call__(self, attn, attn_overlay, bboxes):
        metrics = {'attn_entropy': self.attn_entropy(attn.reshape(-1)), 'no_attn_weight': self.no_attn_weight(attn.reshape(-1))}
        segmentation_label = sent_bboxes_to_segmentation_label(attn_overlay.shape, bboxes)
        for k, v in self.attn_bbox_metrics.items():
            if segmentation_label.sum() > 0:
                metrics[k] = v(attn_overlay.reshape(-1), segmentation_label.reshape(-1).long())
            else:
                metrics[k] = None
        total = np.prod(segmentation_label.shape)
        targets = segmentation_label.reshape(-1).long()
        for p in self.percentile_thresholds:
            if segmentation_label.sum() > 0:
                top_k = int(total * p)
                preds = attn_overlay.reshape(-1)
                threshold = torch.topk(preds, total - top_k, largest=False).values.max()
                pr, re = precision_recall(preds, targets, threshold=threshold)
                f = f1(preds, targets, threshold=threshold)
                iou = ((preds > threshold) & (targets == 1)).float().sum(-1) / \
                      ((preds > threshold) | (targets == 1)).float().sum(-1)
                metrics['precision_at_%f' % p] = pr
                metrics['recall_at_%f' % p] = re
                metrics['f1_at_%f' % p] = f
                metrics['iou_at_%f' % p] = iou
            else:
                metrics['precision_at_%f' % p] = None
                metrics['recall_at_%f' % p] = None
                metrics['f1_at_%f' % p] = None
                metrics['iou_at_%f' % p] = None
        return metrics


def attn_overlay_func(image_shape, attention, bboxes):
    attention_overlay = []
    for attn, (x1, y1, x2, y2) in zip(attention, bboxes):
        attn_overlay = np.zeros((1, *image_shape))
        attn_overlay[:, x1:x2, y1:y2] = attn
        attention_overlay.append(attn_overlay)
    attention_overlay = np.concatenate(attention_overlay, axis=0)
    attention_overlay = attention_overlay.max(axis=0)
    return torch.tensor(attention_overlay)


if __name__ == '__main__':
    data_type = 'randbboxes'
    info_dir = '/home/jered/Documents/data/uniter_data/imagenome/%s_info/test' % data_type
    raw_output_directory = '/home/jered/Documents/data/uniter_data/imagenome/%s_output_test_raw' % data_type
    output_directory = '/home/jered/Documents/data/uniter_data/imagenome/%s_output_test' % data_type
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    if not os.path.exists(os.path.join(output_directory, 'attn')):
        os.mkdir(os.path.join(output_directory, 'attn'))
    if not os.path.exists(os.path.join(output_directory, 'pr_curve')):
        os.mkdir(os.path.join(output_directory, 'pr_curve'))
    if not os.path.exists(os.path.join(output_directory, 'roc_curve')):
        os.mkdir(os.path.join(output_directory, 'roc_curve'))
    rows = []
    metrics_func = Metrics()
    files = os.listdir(raw_output_directory)
    for file in tqdm(files, total=len(files)):
        x = np.load(os.path.join(raw_output_directory, file), allow_pickle=True)
        id_ = '.'.join(file.split('.')[:-1])
        with open(os.path.join(info_dir, 'imagenome_info_%s.json' % id_), 'r') as f:
            info = json.load(f)
        attn = x['attention']
        np.save(os.path.join(output_directory, 'attn', info['dicom_sent_id'] + '.npy'), attn)
        original_shape = eval(info['original_shape'])
        attn_bboxes = process_bboxes(
            [original_shape for _ in x['norm_bb']],
            [[
                int(normalized_bb[0] * original_shape[0]),
                int(normalized_bb[1] * original_shape[1]),
                int(normalized_bb[2] * original_shape[0]),
                int(normalized_bb[3] * original_shape[1]),
            ] for normalized_bb in x['norm_bb']],
        )
        image_shape = (224, 224)
        attn_overlay = attn_overlay_func(image_shape, x['attention'], attn_bboxes)
        metrics = metrics_func(torch.tensor(attn), attn_overlay, eval(info['bboxes']))
        np.save(os.path.join(output_directory, 'pr_curve', info['dicom_sent_id'] + '.npy'),
                np.array(metrics['pr_curve'], dtype=object))
        np.save(os.path.join(output_directory, 'roc_curve', info['dicom_sent_id'] + '.npy'),
                np.array(metrics['roc_curve'], dtype=object))
        del metrics['pr_curve']
        del metrics['roc_curve']
        global_sim = x['scores'][1] - x['scores'][0]
        rows.append({
            'dicom_sent_id': info['dicom_sent_id'],
            'patient_id': info['patient_id'],
            'study_id': info['study_id'],
            'dicom_id': info['dicom_id'],
            'sent_id': info['sent_id'],
            'sentence': info['sentence'],
            'bbox_names': info['bbox_names'],
            'sent_labels': info['sent_labels'],
            'sent_contexts': info['sent_contexts'],
            'bboxes': info['bboxes'],
            'attn_bboxes': attn_bboxes,
            'global_sims': global_sim.item(),
        })
        for k, v in metrics.items():
            if v is not None:
                rows[-1][k] = v.item()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_directory, 'sentences.csv'), index=False)

"""
dicom_sent_id,patient_id,study_id,dicom_id,sent_id,sentence,bbox_names,sent_labels,sent_contexts,bboxes,
auroc,avg_precision,attn_entropy,no_attn_weight,local_sims,global_sims,precision_at_0.050000,recall_at_0.050000,
f1_at_0.050000,iou_at_0.050000,precision_at_0.100000,recall_at_0.100000,f1_at_0.100000,iou_at_0.100000,
precision_at_0.200000,recall_at_0.200000,f1_at_0.200000,iou_at_0.200000,precision_at_0.300000,recall_at_0.300000,
f1_at_0.300000,iou_at_0.300000
"""
