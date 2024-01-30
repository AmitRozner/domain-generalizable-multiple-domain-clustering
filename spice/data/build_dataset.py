import os
import numpy as np
import torchvision
from spice.data.stl10 import STL10
from spice.data.transformations import get_train_transformations
from spice.data.stl10_embedding import STL10EMB
from spice.data.cifar import CIFAR10, CIFAR20
from spice.data.imagenet import ImageNetSubEmb, ImageNetSubEmbLMDB, TImageNetEmbLMDB
from spice.data.npy import NPYEMB
from fixmatch.datasets.dataset import ConcatDatasetWithIdx
import glob

def build_dataset(data_cfg, train_trans1=None, train_trans2=None, balance_datasets=False, soft_balance=False):
    type = data_cfg.type

    dataset = None

    if train_trans1 is None:
        train_trans1 = get_train_transformations(data_cfg.trans1)

    if train_trans2 is None:
        train_trans2 = get_train_transformations(data_cfg.trans2)

    if type == "stl10":
        dataset = STL10(root=data_cfg.root_folder,
                        split=data_cfg.split,
                        show=data_cfg.show,
                        transform1=train_trans1,
                        transform2=train_trans2,
                        download=False)
    elif type == "stl10_emb":
        dataset = STL10EMB(root=data_cfg.root_folder,
                           split=data_cfg.split,
                           show=data_cfg.show,
                           transform1=train_trans1,
                           transform2=train_trans2,
                           download=False,
                           embedding=data_cfg.embedding)
    elif type == "npy_emb":
        dataset = NPYEMB(root=data_cfg.root,
                         show=data_cfg.show,
                         transform1=train_trans1,
                         transform2=train_trans2,
                         embedding=data_cfg.embedding)
    elif type == "cifar10":
        dataset = CIFAR10(root=data_cfg.root_folder,
                          all=data_cfg.all,
                          train=data_cfg.train,
                          transform1=train_trans1,
                          transform2=train_trans2,
                          target_transform=None,
                          download=False,
                          embedding=data_cfg.embedding,
                          show=data_cfg.show,
                          )
    elif type == "cifar100":
        dataset = CIFAR20(root=data_cfg.root_folder,
                          all=data_cfg.all,
                          train=data_cfg.train,
                          transform1=train_trans1,
                          transform2=train_trans2,
                          target_transform=None,
                          download=False,
                          embedding=data_cfg.embedding,
                          show=data_cfg.show,
                          )
    elif type == 'imagenet':
        dataset = ImageNetSubEmb(subset_file=data_cfg.subset_file,
                                 embedding=data_cfg.embedding,
                                 split=data_cfg.split,
                                 transform1=train_trans1,
                                 transform2=train_trans2)
    elif type == 'imagenet_lmdb':
        dataset = ImageNetSubEmbLMDB(lmdb_file=data_cfg.lmdb_file,
                                     meta_info_file=data_cfg.meta_info_file,
                                     embedding=data_cfg.embedding,
                                     split=data_cfg.split,
                                     transform1=train_trans1,
                                     transform2=train_trans2,
                                     resize=data_cfg.resize)
    elif type == 'timagenet_lmdb':
        dataset = TImageNetEmbLMDB(lmdb_file=data_cfg.lmdb_file,
                                   meta_info_file=data_cfg.meta_info_file,
                                   embedding=data_cfg.embedding,
                                   transform1=train_trans1,
                                   transform2=train_trans2)
    elif type.startswith('office') or type.startswith('mnist') or type.startswith('domainnet') or type.startswith('pacs'):
        if isinstance(train_trans1, str) and train_trans1.endswith("_none"):
            train_trans1 = None
            train_trans2 = None

        concat_ds = []
        domain_names_sep = data_cfg.domain_names.split("_")

        if len(domain_names_sep) < 1:
            raise f"--domain_names has to include at least one of {type} data domains to work"

        for ds_name in domain_names_sep:
            if type.startswith('mnist'):
                if ds_name == 'mnistm':
                    from datasets.mnist.mnistm.mnist_m import MNISTM
                    concat_ds.append(MNISTM(root=os.path.join(data_cfg.root_folder, ds_name)))
                elif ds_name == 'mnist':
                    from datasets.mnist.MNIST.rbgmnist import MNISTRGB
                    concat_ds.append(MNISTRGB(root=os.path.join(data_cfg.root_folder, ds_name), download=True))
                else:
                    raise f"Dataset {ds_name} is not implemented"
            else:
                concat_ds.append(torchvision.datasets.ImageFolder(root=os.path.join(data_cfg.root_folder, ds_name)))
        if data_cfg.domains2comparewith != "":
            domains2compete = data_cfg.domains2comparewith.split("_")
            for domain in domain_names_sep:
                if domain  not in domains2compete:
                    domain_added=domain
            dict_num_samples={}
            sum_imgs_tot = 0
            for ds_name in domains2compete:
                ds_path = os.path.join(data_cfg.root_folder, ds_name)
                sum_imgs=0
                for cls in glob.glob(ds_path+"/*"):
                    sum_imgs +=  len(glob.glob(cls+"/*.png"))+len(glob.glob(cls+"/*.jpg"))
                sum_imgs_tot = sum_imgs_tot+sum_imgs
                dict_num_samples[ds_name]=sum_imgs
            ds_path = os.path.join(data_cfg.root_folder, domain_added)
            sum_imgs_added=0
            for cls in glob.glob(ds_path+"/*"):
                sum_imgs_added +=  len(glob.glob(cls+"/*.png"))+len(glob.glob(cls+"/*.jpg"))
            dict_num_samples[domain_added]=sum_imgs_added
            sorted_dict_num_samples = {k: v for k, v in sorted(dict_num_samples.items(), key=lambda item: item[1])}
            avg_samples_per_domain = (sum_imgs_tot)//3+3
            extra=0
            for k,v in sorted_dict_num_samples.items():
                if v+extra<avg_samples_per_domain:
                    extra =extra + abs(v-avg_samples_per_domain)
                else:
                    sorted_dict_num_samples[k]=avg_samples_per_domain+extra
                    extra=0
            for j,ds in enumerate(concat_ds):
                ds_name = ds.root.split("/")[-1]
                num_samples_new_ds = sorted_dict_num_samples[ds_name]
                if num_samples_new_ds==len(ds.samples):
                    continue
                choosen_idxes = np.random.randint(0, high=len(ds.samples), size=num_samples_new_ds)
                new_samples = []
                for sam in choosen_idxes:
                    new_samples.append(ds.samples[sam])
                concat_ds[j].samples  = new_samples
        concat_ds = list(concat_ds)
        num_domains = len(concat_ds)

        if type.endswith('_emb'):
            dataset = ConcatDatasetWithIdx(concat_ds, embeddings=data_cfg.embedding, num_datasets=num_domains,
                                           transform1=train_trans1, transform2=train_trans2,
                                           balance_datasets=balance_datasets, soft_balance=soft_balance)
        else:
            dataset = ConcatDatasetWithIdx(concat_ds, num_datasets=num_domains, transform1=train_trans1,
                                           transform2=train_trans2,  balance_datasets=balance_datasets,
                                           soft_balance=soft_balance)

        dataset.num_classes = data_cfg.num_cluster
    else:
        assert TypeError

    return dataset
