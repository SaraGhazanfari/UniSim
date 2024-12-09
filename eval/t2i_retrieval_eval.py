import json
import pickle
from argparse import ArgumentParser
from typing import List, Dict, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from eval.utils import get_any_model, parse_args
from data.retrieval.circo import CIRCODataset
torch.multiprocessing.set_sharing_strategy('file_system')


def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

@torch.no_grad()
def extract_image_features(dataset: Dataset, clip_model: CLIP, batch_size=32,
                           num_workers=10, device='cuda:0') -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    # Create data loader
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    index_features = []
    index_names = []
    try:
        print(f"extracting image features {dataset.__class__.__name__} - {dataset.split}")
    except Exception as e:
        pass

    # Extract features
    for batch in tqdm(loader):
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            images = batch.get('reference_image')
        if names is None:
            names = batch.get('reference_name')

        images = images.to(device)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features.append(batch_features.cpu())
            index_names.extend(names)
    index_features = torch.vstack(index_features)
    return index_features, index_names


@torch.no_grad()
def circo_generate_val_predictions(clip_model: CLIP, relative_val_dataset: Dataset, 
                                   tokenizer, device='cuda:0') -> Tuple[
    torch.Tensor, List[str], list]:
    """
    Generates features predictions for the validation set of CIRCO
    """

    # Create the data loader
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=10,
                                     pin_memory=False, collate_fn=collate_fn, 
                                     shuffle=False)

    predicted_features_list, image_predicted_features_list = [], []
    target_names_list = []
    gts_img_ids_list = []

    # Compute the features
    for batch in tqdm(relative_val_loader):
        reference_names = batch['reference_name']
        reference_image = batch['reference_image']
        target_names = batch['target_name']
        relative_captions = batch['relative_caption']
        gt_img_ids = batch['gt_img_ids']
        # input_captions = [f"a photo of $ that {caption}" for caption in relative_captions]
        image_predicted_features_list.append(clip_model.encode_image(reference_image.to(device)))
        predicted_features = clip_model.encode_text(tokenizer(relative_captions).to(device))
        print('FEATURES:', image_predicted_features_list[-1].shape, predicted_features.shape)
        gt_img_ids = np.array(gt_img_ids).T.tolist()
        predicted_features_list.append(predicted_features)
        
        
        target_names_list.extend(target_names)
        gts_img_ids_list.extend(gt_img_ids)

    predicted_features = torch.vstack(predicted_features_list)
    image_predicted_features = torch.vstack(predicted_features_list)
    return predicted_features, image_predicted_features, target_names_list, gts_img_ids_list


@torch.no_grad()
def circo_compute_val_metrics(relative_val_dataset: Dataset, clip_model: CLIP, 
                              index_features: torch.Tensor, index_names: List[str],
                              tokenizer, device='cuda:0') \
        -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRCO validation set given the dataset, pseudo tokens and the reference names
    """

    # Generate the predicted features
    predicted_features, image_predicted_features, target_names, gts_img_ids = circo_generate_val_predictions(clip_model, relative_val_dataset, tokenizer)
    ap_at5 = []
    ap_at10 = []
    ap_at25 = []
    ap_at50 = []

    recall_at5 = []
    recall_at10 = []
    recall_at25 = []
    recall_at50 = []

    # Move the features to the device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Normalize the features
    index_features = F.normalize(index_features.float())

    for predicted_feature, image_predicted_feature, target_name, gt_img_ids in tqdm(zip(predicted_features, image_predicted_features, target_names, gts_img_ids)):
        gt_img_ids = np.array(gt_img_ids)[
            np.array(gt_img_ids) != '']  # remove trailing empty strings added for collate_fn
        
        similarity = image_predicted_feature @ index_features.T
        sorted_indices = torch.topk(similarity, dim=-1, k=100).indices.cpu()
        similarity = predicted_feature @ index_features[sorted_indices].T
        sorted_index_names = np.array(index_names)[sorted_indices]
        map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

        ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
        ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
        ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
        ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

        assert target_name == gt_img_ids[0], f"Target name not in GTs {target_name} {gt_img_ids}"
        single_gt_labels = torch.tensor(sorted_index_names == target_name)
        recall_at5.append(float(torch.sum(single_gt_labels[:5])))
        recall_at10.append(float(torch.sum(single_gt_labels[:10])))
        recall_at25.append(float(torch.sum(single_gt_labels[:25])))
        recall_at50.append(float(torch.sum(single_gt_labels[:50])))

    map_at5 = np.mean(ap_at5) * 100
    map_at10 = np.mean(ap_at10) * 100
    map_at25 = np.mean(ap_at25) * 100
    map_at50 = np.mean(ap_at50) * 100
    recall_at5 = np.mean(recall_at5) * 100
    recall_at10 = np.mean(recall_at10) * 100
    recall_at25 = np.mean(recall_at25) * 100
    recall_at50 = np.mean(recall_at50) * 100

    return {
        'circo_map_at5': map_at5,
        'circo_map_at10': map_at10,
        'circo_map_at25': map_at25,
        'circo_map_at50': map_at50,
        'circo_recall_at5': recall_at5,
        'circo_recall_at10': recall_at10,
        'circo_recall_at25': recall_at25,
        'circo_recall_at50': recall_at50,
    }


@torch.no_grad()
def circo_val_retrieval(dataset_path: str, img_path, model, tokenizer, preprocess: callable) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRCO validation set given the pseudo tokens and the reference names
    """
    model = model.float().eval().requires_grad_(False)

    # Extract the index features
    classic_val_dataset = CIRCODataset(img_path, dataset_path, 'val', 'classic', preprocess)
    index_features, index_names = extract_image_features(classic_val_dataset, model)

    # Define the relative validation dataset
    relative_val_dataset = CIRCODataset(img_path, dataset_path, 'val', 'relative', preprocess)

    return circo_compute_val_metrics(relative_val_dataset, model, index_features, 
                                     index_names, tokenizer=tokenizer)
                                     


def main(args):
    tokenizer, model, preprocess, _ = get_any_model(args)
    circo_metrics = circo_val_retrieval(img_path='/coco/', dataset_path="/vast/sg7457/uni_data/CIRCO",
                                        model=model, preprocess=preprocess, tokenizer=tokenizer)

    for k, v in circo_metrics.items():
        print(f"{k} = {v:.2f}")


if __name__ == '__main__':
    args = parse_args()
    main(args)