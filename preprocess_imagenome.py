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
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image


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


def bbox_to_mask(bbox, image_shape):
    box_mask = torch.zeros(image_shape, dtype=torch.bool)
    box_mask[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1] = 1
    return box_mask


def normalize(image):
    image = image.float()
    return ((image - image.min()) / (image.max() - image.min()))


def original_tensor_to_numpy_image(image):
    return np.array(normalize(image) * 255, dtype=np.uint8)


def resize_img(img, scale):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=cv2.INTER_AREA
    )  # this flips the desireable_size vector

    # Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(
        resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
    )

    return resized_img


def build_transformation():
    t = [transforms.ToTensor(), transforms.CenterCrop(224), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(t)


transform = build_transformation()


def process_img(images, device):
    all_imgs = []
    for x in images:
        # tranform images
        x = resize_img(x, 256)
        img = Image.fromarray(x).convert("RGB")
        img = transform(img)
        all_imgs.append(img)
    all_imgs = torch.stack(all_imgs).to(device)
    return all_imgs


def mask_to_bbox(box_mask):
    if box_mask.sum() == 0:
        return [-1, -1, -1, -1]
    indices0 = torch.arange(box_mask.shape[0])
    indices1 = torch.arange(box_mask.shape[1])
    indices0 = indices0.unsqueeze(1).expand(*box_mask.shape)[box_mask]
    indices1 = indices1.unsqueeze(0).expand(*box_mask.shape)[box_mask]
    return [indices1.min().item(), indices0.min().item(), indices1.max().item(), indices0.max().item()]


def process_bboxes(image_shapes, bboxes):
    box_masks = []
    for shape, bbox in zip(image_shapes, bboxes):
        box_mask = bbox_to_mask(bbox, shape)
        box_masks.append(original_tensor_to_numpy_image(box_mask))
    new_box_masks = process_img(box_masks, 'cpu')
    new_box_masks = new_box_masks > 0
    new_bboxes = [mask_to_bbox(new_box_mask[0]) for new_box_mask in new_box_masks]
    return new_bboxes


def process_instance(args):
    instance, txt_dir, info_dir, img_dir, npz_dir, gold_coords_df, is_gold = args
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
    info_fname = os.path.join(
        info_dir, 'imagenome_info_%s.json' % identifier)
    image_fname = os.path.join(
        img_dir, 'dicom_%s_sent_%s.npy' % (dicom_id, sent_id))
    if info_dir is not None and not os.path.exists(info_fname):
        image_tensor = instance[patient_id][study_id]['images'][dicom_id]
        reshaped_image = process_img([original_tensor_to_numpy_image(image_tensor)], 'cpu')[0, 0]
        np.save(image_fname, reshaped_image)
        objects = instance[patient_id][study_id]['objects'][dicom_id]
        sent_info = objects['sent_to_bboxes'][sent_id]
        bboxes = sent_info['coords_original']
        original_image_shapes = [image_tensor.shape for bbox in bboxes]
        new_bboxes = process_bboxes(original_image_shapes, bboxes)
        with open(info_fname, 'w') as f:
            json.dump({
                'dicom_sent_id': 'dicom_%s_sent_%s' % (dicom_id, sent_id),
                'patient_id': patient_id,
                'study_id': study_id,
                'dicom_id': dicom_id,
                'sent_id': sent_id,
                'sentence': sentence,
                'bbox_names': str(sent_info['bboxes']),
                'sent_labels': str(sent_info['labels']),
                'sent_contexts': str(sent_info['contexts']),
                'original_shape': str(list(image_tensor.shape)),
                'bboxes': str(new_bboxes),
            }, f)
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


def get_gen(dl, name, txt_dir, info_dir, img_dir, npz_dir, gold_coords_df, gold_test):
    # num_images_per_folder = 50000
    for i, instance in tqdm(enumerate(dl), total=len(dl)):
        # folder = os.path.join(jpg_dir, 'folder%i' % (i // num_images_per_folder))
        # if not os.path.exists(folder):
        #     os.mkdir(folder)
        yield instance, txt_dir, info_dir, img_dir, npz_dir, gold_coords_df, gold_test and name == 'test'


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    data_type = 'gensentswapcond'
    dm_kwargs = dict(
        generate_sent=True, swap_conditions=True
    )
    txt_dir = '/scratch/mcinerney.de/uniter_data/imagenome/%s_text/' % data_type
    info_dir = '/scratch/mcinerney.de/uniter_data/imagenome/%s_info/' % data_type
    img_dir = '/scratch/mcinerney.de/uniter_data/imagenome/%s_image/' % data_type
    npz_dir = '/scratch/mcinerney.de/uniter_data/imagenome/%s_npz/' % data_type
    gold_test = True
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)
    if not os.path.exists(info_dir):
        os.mkdir(info_dir)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    if not os.path.exists(npz_dir):
        os.mkdir(npz_dir)
    mimic_cxr_filer = MimicCxrFiler(download_directory='/scratch/mcinerney.de/mimic-cxr')
    imagenome_filer = ImaGenomeFiler(download_directory='/scratch/mcinerney.de/imagenome')
    print(class2index)
    print(len(class2index), 'classes')
    dm = ImaGenomeDataModule(
        mimic_cxr_filer, imagenome_filer, split_slices='', parallel=True, gold_test=gold_test,
        **dm_kwargs
    )
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
    for name, dataset, indices in name_dl_s[:1]:
        print(name)
        if os.path.exists(os.path.join(txt_dir, name)):
            print('ignoring previously saved data')
            files = os.listdir(os.path.join(txt_dir, name))
            existing_indices = set([int(fname.split('_')[2]) for fname in tqdm(files, total=len(files))])
            indices = list(set(indices).difference(existing_indices))
        dataset = Subset(dataset, sorted(indices))
        dl = DataLoader(dataset, num_workers=8, batch_size=1, collate_fn=lambda b: b[0])
        if not os.path.exists(os.path.join(txt_dir, name)):
            os.mkdir(os.path.join(txt_dir, name))
        if not os.path.exists(os.path.join(info_dir, name)):
            os.mkdir(os.path.join(info_dir, name))
        if not os.path.exists(os.path.join(img_dir, name)):
            os.mkdir(os.path.join(img_dir, name))
        if not os.path.exists(os.path.join(npz_dir, name)):
            os.mkdir(os.path.join(npz_dir, name))
        with mp.Pool(40) as p:
            generator = get_gen(
                dl, name, os.path.join(txt_dir, name), os.path.join(info_dir, name), os.path.join(img_dir, name),
                os.path.join(npz_dir, name),
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
