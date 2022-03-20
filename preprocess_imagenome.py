from imagenome import ImaGenomeDataModule, MimicCxrFiler, ImaGenomeFiler
import numpy as np
from argparse import ArgumentParser
import cv2
import os
from tqdm import tqdm
import multiprocessing as mp
import json
import pandas as pd
import torchvision
import torch
from torch.utils.data import DataLoader, Subset


classes = [
    'left lung',
    'right lung',
    'cardiac silhouette',
    'mediastinum',
    'left lower lung zone',
    'right lower lung zone',
    'right hilar structures',
    'left hilar structures',
    'upper mediastinum',
    'left costophrenic angle',
    'right costophrenic angle',
    'left mid lung zone',
    'right mid lung zone',
    'aortic arch',
    'right upper lung zone',
    'left upper lung zone',
    'right hemidiaphragm',
    'right clavicle',
    'left clavicle',
    'left hemidiaphragm',
    'right apical zone',
    'trachea',
    'left apical zone',
    'carina',
    'svc',
    'right atrium',
    'cavoatrial junction',
    'abdomen',
    'spine',
    'descending aorta',
    'left cardiac silhouette',
    'right cardiac silhouette',
    'left cardiophrenic angle',
    'right cardiophrenic angle',
    'left upper abdomen',
    'right upper abdomen'
]
class2index = {c: i for i, c in enumerate(sorted(classes))}


def get_features(image_tensor, dicom_id, imagenome_filer, gold_coords_df, is_gold):
    bboxes = {}
    if is_gold:
        rows = gold_coords_df[gold_coords_df.image_id == ('%s.dcm' % dicom_id)]
        for i, row in rows.iterrows():
            bboxes[row.bbox_name] = [row.original_x1, row.original_y1, row.original_x2, row.original_y2]
    else:
        scene_graph = imagenome_filer.get_silver_scene_graph_json(dicom_id)
        for object in scene_graph['objects']:
            bboxes[object['bbox_name']] = [
                object['original_x1'], object['original_y1'], object['original_x2'], object['original_y2']]
    new_bboxes = []
    features = []
    names = []
    w, h = image_tensor.shape
    for k in sorted(list(bboxes.keys())):
        x1, y1, x2, y2 = bboxes[k]
        x1 = min(max(x1, 0), w)
        y1 = min(max(y1, 0), h)
        x2 = min(max(x2, 0), w)
        y2 = min(max(y2, 0), h)
        box = image_tensor[x1:x2, y1:y2]
        if np.prod(box.shape) == 0:
            continue
        newdim = 45
        new_box = torchvision.transforms.Resize((newdim, newdim))(box.unsqueeze(0).unsqueeze(0))
        box_features = torch.cat([new_box.reshape(-1), torch.zeros(2048 - newdim * newdim)], 0)
        names.append(k)
        new_bboxes.append(np.array([x1, y1, x2, y2]))
        features.append(box_features.numpy())
    return names, np.stack(new_bboxes, axis=0), np.stack(features, axis=0)


def get_normalized_bboxes(bboxes, image_w, image_h):
    box_width = bboxes[:, 2] - bboxes[:, 0]
    box_height = bboxes[:, 3] - bboxes[:, 1]
    scaled_width = box_width / image_w
    scaled_height = box_height / image_h
    scaled_x = bboxes[:, 0] / image_w
    scaled_y = bboxes[:, 1] / image_h

    scaled_width = scaled_width[..., np.newaxis]
    scaled_height = scaled_height[..., np.newaxis]
    scaled_x = scaled_x[..., np.newaxis]
    scaled_y = scaled_y[..., np.newaxis]

    normalized_bbox = np.concatenate((scaled_x, scaled_y,
                                      scaled_x + scaled_width,
                                      scaled_y + scaled_height,
                                      scaled_width, scaled_height), axis=1)
    return normalized_bbox


def process_instance(args):
    instance, txt_dir, jpg_dir, npz_dir, gold_coords_df, is_gold = args
    patient_id = next(iter(instance.keys()))
    study_id = next(iter(instance[patient_id].keys()))
    dicom_id = next(iter(instance[patient_id][study_id]['images'].keys()))
    sent_id = instance[patient_id][study_id]['sent_id']
    sentence = instance[patient_id][study_id]['sentence']
    identifier = '%s_%s' % (dicom_id, sent_id)
    # img_fname = os.path.join(jpg_dir, 'imagenome_img_%s.jpg' % dicom_id)
    # if not os.path.exists(img_fname):
    #     image_tensor = instance[patient_id][study_id]['images'][dicom_id]
    #     cv2.imwrite(img_fname, image_tensor.numpy())
    fea_fname = os.path.join(npz_dir, 'imagenome_fea_%s.npz' % dicom_id)
    if not os.path.exists(fea_fname):
        image_tensor = instance[patient_id][study_id]['images'][dicom_id]
        image_w, image_h = image_tensor.shape
        names, bboxes, features = get_features(image_tensor, dicom_id, imagenome_filer, gold_coords_df, is_gold)
        normalized_bbox = get_normalized_bboxes(bboxes, image_w, image_h)
        soft_labels = np.zeros((features.shape[0], len(class2index)))
        soft_labels[np.arange(features.shape[0]), np.array([class2index[n] for n in names])] = 1.
        np.savez_compressed(fea_fname,
                            norm_bb=normalized_bbox.astype(np.float16),
                            features=features.astype(np.float16),
                            conf=np.ones(features.shape[0]).astype(np.float16),
                            soft_labels=soft_labels.astype(np.float16))
    txt_fname = os.path.join(
        txt_dir, 'imagenome_txt_%i_%s.json' % (instance[patient_id][study_id]['index'], identifier))
    if not os.path.exists(txt_fname):
        with open(txt_fname, 'w') as f:
            json.dump({'identifier': identifier, 'sentence': sentence}, f)


def first_n(gen, n):
    for i, item in enumerate(gen):
        if i < n:
            yield item
        else:
            break


def get_gen(dl, name, txt_dir, jpg_dir, npz_dir, gold_coords_df, gold_test):
    num_images_per_folder = 50000
    for i, instance in tqdm(enumerate(dl), total=len(dl)):
        folder = os.path.join(jpg_dir, 'folder%i' % (i // num_images_per_folder))
        if not os.path.exists(folder):
            os.mkdir(folder)
        yield instance, txt_dir, folder, npz_dir, gold_coords_df, gold_test and name == 'test'


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    txt_dir = '/scratch/mcinerney.de/uniter_data/imagenome/normal_text/'
    jpg_dir = '/scratch/mcinerney.de/uniter_data/imagenome/normal_jpgs/'
    npz_dir = '/scratch/mcinerney.de/uniter_data/imagenome/normal_npz/'
    gold_test = True
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)
    if not os.path.exists(jpg_dir):
        os.mkdir(jpg_dir)
    if not os.path.exists(npz_dir):
        os.mkdir(npz_dir)
    mimic_cxr_filer = MimicCxrFiler(download_directory='/scratch/mcinerney.de/mimic-cxr')
    imagenome_filer = ImaGenomeFiler(download_directory='/scratch/mcinerney.de/imagenome')
    print(class2index)
    print(len(class2index), 'classes')
    dm = ImaGenomeDataModule(
        mimic_cxr_filer, imagenome_filer, split_slices='', parallel=True, gold_test=gold_test)
    # dm.train_dataloader_kwargs['shuffle'] = False
    # sl = slice(0, 100000)
    # sl = slice(100000, 200000)
    # sl = slice(200000, 300000)
    # sl = slice(300000, 400000)
    # sl = slice(400000, 500000)
    # sl = slice(500000, 600000)
    # sl = slice(600000, 700000)
    # sl = slice(700000, 800000)
    # sl = slice(800000, 900000)
    name_dl_s = [
        ('test', dm.test, list(range(len(dm.test)))),
        ('val', dm.val, list(range(len(dm.val)))),
        ('train', dm.train, list(range(len(dm.train)))[:])]
    for name, dataset, indices in name_dl_s[:]:
        print(name)
        if os.path.exists(os.path.join(txt_dir, name)):
            print('ignoring previously saved data')
            files = os.listdir(os.path.join(txt_dir, name))
            existing_indices = set([int(fname.split('_')[2]) for fname in tqdm(files, total=len(files))])
            indices = list(set(indices).difference(existing_indices))
        dataset = Subset(dataset, sorted(indices))
        dl = DataLoader(dataset, num_workers=32, batch_size=1, collate_fn=lambda b: b[0])
        if not os.path.exists(os.path.join(txt_dir, name)):
            os.mkdir(os.path.join(txt_dir, name))
        if not os.path.exists(os.path.join(jpg_dir, name)):
            os.mkdir(os.path.join(jpg_dir, name))
        if not os.path.exists(os.path.join(npz_dir, name)):
            os.mkdir(os.path.join(npz_dir, name))
        with mp.Pool(16) as p:
            generator = get_gen(
                dl, name, os.path.join(txt_dir, name), os.path.join(jpg_dir, name), os.path.join(npz_dir, name),
                pd.read_csv(
                    imagenome_filer.get_full_path('gold_dataset/auto_bbox_pipeline_coordinates_1000_images.txt'),
                    delimiter='\t'),
                gold_test
            )
            x = p.imap_unordered(
                process_instance,
                generator
            )
            for _ in tqdm(x, total=len(dl)):
                pass
