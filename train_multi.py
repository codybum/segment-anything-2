# Train/Fine-Tune SAM 2 on the LabPics 1 dataset
import argparse

# Toturial: https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3
# Main repo: https://github.com/facebookresearch/segment-anything-2
# Labpics Dataset can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1
# Pretrained models for sam2 Can be downloaded from: https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints

import numpy as np
import torch
import cv2
import os

from datasets import LabPicsV1
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor_multi import SAM2ImagePredictor

#for dist
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def collate_fn(batch):
    return tuple(zip(*batch))

def train(rank, args):

    print(f"Running DDP Training on rank {rank}.")
    setup(rank, args.world_size)

    # Load model
    sam2_checkpoint = "../models/sam2-hiera-small/sam2_hiera_small.pt"  # path to model weight (pre model loaded from: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt)
    model_cfg = "sam2_hiera_s.yaml"  # model config
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=rank)  # load model

    #DDP
    sam2_model_ddp = DDP(sam2_model, device_ids=[rank])

    predictor = SAM2ImagePredictor(sam2_model_ddp)

    # Set training parameters
    predictor.model.module.sam_mask_decoder.train(True)  # enable training of mask decoder
    predictor.model.module.sam_prompt_encoder.train(True)  # enable training of prompt encoder
    optimizer = torch.optim.AdamW(params=predictor.model.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()  # mixed precision

    #training data

    class PadSequence:
        def __call__(self, batch):
            # Let's assume that each element in "batch" is a tuple (data, label).
            # Sort the batch in the descending order
            sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
            # Get each sequence and pad it
            sequences = [x[0] for x in sorted_batch]
            sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            # Also need to store the length of each sequence
            # This is later needed in order to unpad the sequences
            lengths = torch.LongTensor([len(x) for x in sequences])

            # Don't forget to grab the labels of the *sorted* batch
            labels = torch.LongTensor(map(lambda x: x[1], sorted_batch))
            return sequences_padded, lengths, labels

    labpicsv1_train_dataset = LabPicsV1(args)
    train_sampler = DistributedSampler(labpicsv1_train_dataset, rank=rank, shuffle=True, seed=args.random_state)
    train_loader = DataLoader(labpicsv1_train_dataset, batch_size=2, num_workers=0,
                              pin_memory=True, sampler=train_sampler)

    # Training loop
    with torch.cuda.amp.autocast():  # cast to mix precision

        itr = 0
        master_process = rank == 0

        for epoch in range(args.num_epochs):

            with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img', disable=not master_process) as pbar:

                #image, mask, input_point, input_label = read_batch(data)  # load data batch

                for ind, pack in enumerate(train_loader):

                    # input image and gt masks
                    image = pack['image'].numpy()
                    mask = pack['mask'].numpy()
                    input_point = pack['input_point'].numpy()
                    input_label = pack['input_label'].numpy()

                    #image, mask, input_point, input_label = read_batch(data)  # load data batch

                    if mask.shape[0] == 0: continue  # ignore empty batches
                    predictor.set_image(image)  # apply SAM image encoder to the image

                    # prompt encoding

                    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None,
                                                                                            mask_logits=None,
                                                                                            normalize_coords=True)
                    sparse_embeddings, dense_embeddings = predictor.model.module.sam_prompt_encoder(points=(unnorm_coords, labels),
                                                                                                    boxes=None, masks=None, )

                    # mask decoder

                    batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                    low_res_masks, prd_scores, _, _ = predictor.model.module.sam_mask_decoder(
                        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                        image_pe=predictor.model.module.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings, multimask_output=True, repeat_image=batched_mode,
                        high_res_features=high_res_features, )
                    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[
                        -1])  # Upscale the masks to the original image resolution

                    # Segmentaion Loss caclulation

                    #gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                    gt_mask = torch.tensor(mask.astype(np.float32)).to(rank)
                    prd_mask = torch.sigmoid(prd_masks[:, 0])  # Turn logit map to probability map
                    seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log(
                        (1 - prd_mask) + 0.00001)).mean()  # cross entropy loss

                    # Score loss calculation (intersection over union) IOU

                    inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                    iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                    loss = seg_loss + score_loss * 0.05  # mix losses

                    # apply back propogation

                    predictor.model.module.zero_grad()  # empty gradient
                    scaler.scale(loss).backward()  # Backpropogate
                    scaler.step(optimizer)
                    scaler.update()  # Mix precision

                    if rank == 0:

                        if itr % 10 == 0:
                            # Display results
                            if itr == 0: mean_iou = 0
                            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
                            #print("step)", itr, "Accuracy(IOU)=", mean_iou)
                            pbar.set_postfix(loss=loss.item(), iou=mean_iou)

                        if itr % 1000 == 0:
                            torch.save(predictor.model.module.state_dict(), args.output_model_path)

                    itr += 1

                    pbar.update()

    cleanup()

def read_batch(data): # read random image and its annotaion from  the dataset (LabPics)

   #  select image

        ent  = data[np.random.randint(len(data))] # choose random entry
        Img = cv2.imread(ent["image"])[...,::-1]  # read image
        ann_map = cv2.imread(ent["annotation"]) # read annotation

   # resize image

        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]]) # scalling factor
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),interpolation=cv2.INTER_NEAREST)

   # merge vessels and materials annotations

        mat_map = ann_map[:,:,0] # material annotation map
        ves_map = ann_map[:,:,2] # vessel  annotaion map
        mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1) # merge maps

   # Get binary masks and points

        inds = np.unique(mat_map)[1:] # load all indices
        points= []
        masks = []
        for ind in inds:
            mask=(mat_map == ind).astype(np.uint8) # make binary mask corresponding to index ind
            masks.append(mask)
            coords = np.argwhere(mask > 0) # get all coordinates in mask
            yx = np.array(coords[np.random.randint(len(coords))]) # choose random point/coordinate
            points.append([[yx[1], yx[0]]])

        return Img, np.array(masks), np.array(points), np.ones([len(masks),1])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multi-GPU Training: SAM2')

    #general
    parser.add_argument('--random_state', type=int, default=42,
                        help='Pass an int for reproducible output across multiple function calls')

    #data management
    parser.add_argument('--input_data_path', type=str, default='../data/LabPicsV1/', help='name of project')
    parser.add_argument('--output_model_path', type=str, default='sam2_checkpoint', help='name of project')
    parser.add_argument('--test_size', type=float, default=0.2, help='size of the testing split')

    #training params
    parser.add_argument('--num_epochs', type=int, default=10, help='name of project')
    parser.add_argument('--world_size', type=int, help='name of project')
    parser.add_argument('--lr', type=float, default=1e-5, help='name of project')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='name of project')

    # get args
    args = parser.parse_args()

    if args.world_size is None:
        if 'WORLD_SIZE' in os.environ:
            args.world_size = int(os.environ['WORLD_SIZE'])
        else:
            args.world_size = torch.cuda.device_count()

    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)