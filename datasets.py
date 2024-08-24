""" train and test dataset

"""
import os

import numpy as np
from torch.utils.data import Dataset
import cv2

class LabPicsV1(Dataset):
    def __init__(self, args):

        #Read data
        self.data = []  # list of files in dataset
        for ff, name in enumerate(os.listdir(args.input_data_path + "Simple/Train/Image/")):  # go over all folder annotation
            self.data.append({"image": args.input_data_path + "Simple/Train/Image/" + name,
                         "annotation": args.input_data_path + "Simple/Train/Instance/" + name[:-4] + ".png"})

        print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):  # read random image and its annotaion from  the dataset (LabPics)

        #  select image

        ent = self.data[index]  # choose random entry
        Img = cv2.imread(ent["image"])[..., ::-1]  # read image
        ann_map = cv2.imread(ent["annotation"])  # read annotation

        # resize image

        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # scalling factor
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                             interpolation=cv2.INTER_NEAREST)

        # merge vessels and materials annotations

        mat_map = ann_map[:, :, 0]  # material annotation map
        ves_map = ann_map[:, :, 2]  # vessel  annotaion map
        mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)  # merge maps

        # Get binary masks and points

        inds = np.unique(mat_map)[1:]  # load all indices
        points = []
        masks = []
        for ind in inds:
            mask = (mat_map == ind).astype(np.uint8)  # make binary mask corresponding to index ind
            masks.append(mask)
            coords = np.argwhere(mask > 0)  # get all coordinates in mask
            yx = np.array(coords[np.random.randint(len(coords))])  # choose random point/coordinate
            points.append([[yx[1], yx[0]]])
        #return Img, np.array(masks), np.array(points), np.ones([len(masks), 1])

        return {
            'image': Img,
            'mask': np.array(masks),
            'input_point': np.array(points),
            'input_label': np.ones([len(masks), 1])
        }



