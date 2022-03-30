from model.pretrain import UniterForPretraining
from utils.const import IMG_DIM, IMG_LABEL_DIM
import torch
from data.itm import ItmDataset, itm_ot_collate
from data.data import DetectFeatLmdb, TxtTokLmdb
from utils.misc import Struct
import json
from horovod import torch as hvd
hvd.init()
import os
from torch.utils.data import DataLoader
from data import PrefetchLoader
from tqdm import tqdm
import numpy as np


def save_img2txts(txt_db):
    with open(os.path.join(txt_db, 'txt2img.json'), 'r') as f:
        txt2img = json.load(f)
    img2txts = {}
    for k, v in txt2img.items():
        if v not in img2txts.keys():
            img2txts[v] = []
        img2txts[v].append(k)
    with open(os.path.join(txt_db, 'img2txts.json'), 'w') as f:
        json.dump(img2txts, f)


class UniterForImageTextAlignment(UniterForPretraining):
    def forward(self, batch):
        input_ids = batch['input_ids']
        img_feat = batch['img_feat']
        ot_inputs = batch['ot_inputs']
        targets = batch['targets']
        all_output, all_attention_probs = self.uniter(
            input_ids, batch['position_ids'],
            img_feat, batch['img_pos_feat'],
            batch['attn_masks'], gather_index=batch['gather_index'], img_masks=None,
            output_all_encoded_layers=True,
            output_attention=True,
            txt_type_ids=None, img_type_ids=None)
        sequence_output = all_output[-1]
        pooled_output = self.uniter.pooler(sequence_output)
        itm_scores = self.itm_output(pooled_output)
        attention_probs = torch.cat(all_attention_probs, 1)
        attention_probs_mean = attention_probs.mean(1)
        b = sequence_output.size(0)
        tl = input_ids.size(1)
        il = img_feat.size(1)
        max_l = max(ot_inputs['scatter_max'] + 1, tl+il)
        ot_scatter = ot_inputs['ot_scatter'].unsqueeze(1).expand_as(attention_probs_mean)
        attention = torch.zeros(
            *attention_probs_mean.shape[:2], max_l,
            dtype=attention_probs_mean.dtype, device=attention_probs_mean.device
        ).scatter_(dim=2, index=ot_scatter, src=attention_probs_mean)
        ot_scatter = ot_inputs['ot_scatter'].unsqueeze(-1).expand_as(attention)
        attention = torch.zeros(
            attention.shape[0], max_l, max_l,
            dtype=attention.dtype, device=attention.device
        ).scatter_(dim=1, index=ot_scatter, src=attention)
        attention = torch.stack([
            attention[:, :tl, tl:tl+il],
            attention[:, tl:tl+il, :tl].transpose(1, 2)], 1)
#         txt_pad = ot_inputs['txt_pad']
#         img_pad = ot_inputs['img_pad']
#         txt_img_pad = (txt_pad.unsqueeze(2) == 1) | (img_pad.unsqueeze(1) == 1)
        attention = attention.mean(1).mean(1)
        return {'pooled_output': pooled_output, 'itm_scores': itm_scores, 'attention': attention}


if __name__ == '__main__':
    data_type = 'gensentswapcond'
    save_to = '/imagenome/%s_output_test_raw' % data_type
    imagenome_info = '/imagenome/%s_info/test' % data_type
    img_db = '/img/imagenome_%s/test' % data_type
    txt_db = '/txt/imagenome_%s/test.db' % data_type
    checkpoint = torch.load('/storage/experiment_2gpu4/ckpt/model_step_200000.pt')
    if not os.path.exists(save_to):
        os.mkdir(save_to)
    if not os.path.exists(os.path.join(txt_db, 'img2txts.json')):
        save_img2txts(txt_db)
    m = UniterForImageTextAlignment.from_pretrained(
        '/src/config/uniter-base.json',
        checkpoint,
        img_dim=IMG_DIM, img_label_dim=IMG_LABEL_DIM)
    m = m.to('cuda')
    train_opts = Struct(json.load(open('/src/config/pretrain-imagenome-base-1gpu.json')))
    eval_img_db = DetectFeatLmdb(img_db,
                                 train_opts.conf_th, train_opts.max_bb,
                                 train_opts.min_bb, train_opts.num_bb,
                                 False)
    eval_txt_db = TxtTokLmdb(txt_db, -1)
    eval_dataset = ItmDataset(eval_txt_db, eval_img_db, 0)
    eval_dataloader = DataLoader(eval_dataset, batch_size=3,
                                 num_workers=1,
                                 pin_memory=True,
                                 collate_fn=itm_ot_collate)
    eval_dataloader = PrefetchLoader(eval_dataloader)
    offset = 0
    attentions = []
    itm_scores = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            outs = m(batch)
            for scores, attn, mask in zip(
                    outs['itm_scores'].cpu(), outs['attention'].cpu(), batch['ot_inputs']['img_pad'].cpu()):
                attentions.append(attn[mask == 0].numpy())
                itm_scores.append(scores.numpy())
    for i in tqdm(range(len(eval_dataset)), total=len(eval_dataset)):
        id_ = eval_dataset.ids[i]
        scores = itm_scores[i]
        attention = attentions[i]
        img_pos_feat = eval_dataset[i][2].numpy()
        np.savez_compressed(os.path.join(save_to, id_ + '.npz'),
                            scores=scores,
                            attention=attention,
                            norm_bb=img_pos_feat[:, :4])

"""
dicom_sent_id,patient_id,study_id,dicom_id,sent_id,sentence,bbox_names,sent_labels,sent_contexts,bboxes,
auroc,avg_precision,attn_entropy,no_attn_weight,local_sims,global_sims,precision_at_0.050000,recall_at_0.050000,
f1_at_0.050000,iou_at_0.050000,precision_at_0.100000,recall_at_0.100000,f1_at_0.100000,iou_at_0.100000,
precision_at_0.200000,recall_at_0.200000,f1_at_0.200000,iou_at_0.200000,precision_at_0.300000,recall_at_0.300000,
f1_at_0.300000,iou_at_0.300000
"""
