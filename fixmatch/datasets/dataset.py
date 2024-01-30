from torchvision import datasets, transforms
from torch.utils.data import Dataset, ConcatDataset
from .data_utils import get_onehot
from .augmentation.randaugment import RandAugment
import bisect
from typing import Iterable
from PIL import Image
import numpy as np
import copy
import torch

class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """
    def __init__(self,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 use_strong_transform=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.data = data
        self.targets = targets
        
        self.num_classes = num_classes
        self.use_strong_transform = use_strong_transform
        self.onehot = onehot
        
        self.transform = transform
        if use_strong_transform:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3,5))
        else:
            self.strong_transform = strong_transform
                
    
    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        
        #set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)
            
        #set augmented images
            
        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.use_strong_transform:
                return img_w, target, idx
            else:
                return img_w, self.strong_transform(img), target

    
    def __len__(self):
        return len(self.data)


class ConcatDatasetWithIdx(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset], embeddings=None, num_datasets=2, transform1=None, transform2=None,
                 balance_datasets=False, soft_balance=False):
        super(ConcatDatasetWithIdx, self).__init__(datasets)
        self.embeddings = np.load(embeddings) if embeddings is not None else None
        self.num_datasets = num_datasets
        self.transform1 = transform1
        self.transform2 = transform2
        self.data_start_ind = [0]
        self.datasets_lengths = []
        self.balance_datasets = balance_datasets
        self.soft_balance = soft_balance
        for ds_ind, dataset in enumerate(datasets):
            self.datasets_lengths.append(len(dataset))

            if ds_ind > 0:
                add_last_ds = len(datasets[ds_ind - 1])
                self.data_start_ind.append(self.data_start_ind[-1] + add_last_ds)


    def get_idxs(self, idx):
        if self.balance_datasets or self.soft_balance:
            sample_idx_vec = idx % np.asarray(self.datasets_lengths)
            return sample_idx_vec
        else:
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx_vec = idx
            else:
                sample_idx_vec = idx - self.cumulative_sizes[dataset_idx - 1]
            return dataset_idx, sample_idx_vec


    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        if self.balance_datasets or self.soft_balance:
            sample_idx = self.get_idxs(idx)
            domain_label = list(range(len(sample_idx)))
            if self.soft_balance:
                chosen_ind = np.random.choice(domain_label, 1)[0]
                domain_label = domain_label[chosen_ind]
                sample_idx = sample_idx[chosen_ind]
                curr_samples = [self.datasets[chosen_ind][sample_idx]]
            else:
                curr_samples = []
                for ds_ind, dataset in enumerate(self.datasets):
                    curr_samples.append(dataset[sample_idx[ds_ind]])
        else:
            domain_label, sample_idx = self.get_idxs(idx)
            curr_samples = [self.datasets[domain_label][sample_idx]]

        img_list = [curr_sample[0] for curr_sample in curr_samples]
        class_label_list = [curr_sample[1] for curr_sample in curr_samples]

        if self.transform1 is not None:
            img_trans1_list = [self.transform1(img) for img in img_list]
        else:
            img_trans1_list = img_list

        if self.transform2 is not None:
            img_trans2_list = [self.transform2(img) for img in img_list]
        else:
            img_trans2_list = img_list

        if self.embeddings is not None:
            if self.balance_datasets and not self.soft_balance:
                embeddings_list = [torch.tensor(self.embeddings[ind,:]) for ind in (self.data_start_ind + sample_idx)]
                return img_trans1_list, img_trans2_list, embeddings_list, class_label_list, domain_label, sample_idx + \
                       self.data_start_ind
            else:
                embeddings_list = [self.embeddings[sample_idx + self.data_start_ind[domain_label], :]]
                return img_trans1_list, img_trans2_list, embeddings_list, class_label_list, [domain_label], sample_idx + \
                       self.data_start_ind[domain_label]
        else:
            if self.balance_datasets and not self.soft_balance:
                return img_trans1_list, img_trans2_list, class_label_list, domain_label, sample_idx + self.data_start_ind
            else:
                return img_trans1_list, img_trans2_list, class_label_list, [domain_label], sample_idx + \
                       self.data_start_ind[domain_label]