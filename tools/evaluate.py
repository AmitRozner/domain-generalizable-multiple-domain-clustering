from __future__ import print_function, division
import os
from moco.builder import FFN
import torch
import glob
import sys
import torchvision
import PIL
import torchvision.transforms.functional as transform
from collections import OrderedDict

sys.path.insert(0, './')
try:
    sys.path.insert(0, './spice/')
except:
    pass
import matplotlib
from fixmatch.utils import net_builder
from fixmatch.datasets.ssl_dataset_robust import SSL_Dataset
from fixmatch.datasets.data_utils import get_data_loader
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari
import warnings
import matplotlib.pyplot as plt
import shutil

warnings.filterwarnings('ignore')
from adain.adain import AdaIN
from eval_kmeans import main as kmeans_evaluation


def internal_read_image(image_path):
    PIL_image = PIL.Image.open(image_path)
    tensor_image = transform.to_tensor(PIL_image)
    return tensor_image


def remove_feature_hirerchy(dict):
    from collections import OrderedDict
    train_dict = OrderedDict()
    eval_dict = OrderedDict()
    domain_dict = OrderedDict()
    for k, v in dict.items():
        if "feature_module" in k:
            new_k = k.replace("feature_module.", "")
            new_k_eval = k.replace("module.feature_module.", "")
        else:
            new_k = k
            new_k_eval = k
        if "classifier" in new_k and "lin" in new_k:
            if "classifier.lin1.weight" in new_k:
                new_k = new_k.replace("module.head.head_0.classifier.lin1.weight", "module.fc.fc1.weight")
                new_k_eval = new_k_eval.replace("module.head.head_0.classifier.lin1.weight", "fc.fc1.weight")

            elif "classifier.lin1.bias" in new_k:
                new_k = new_k.replace("module.head.head_0.classifier.lin1.bias", "module.fc.fc1.bias")
                new_k_eval = new_k_eval.replace("module.head.head_0.classifier.lin1.bias", "fc.fc1.bias")
            elif "classifier.lin2.weight" in new_k:
                new_k = new_k.replace("module.head.head_0.classifier.lin2.weight", "module.fc.fc2.weight")
                new_k_eval = new_k_eval.replace("module.head.head_0.classifier.lin2.weight", "fc.fc2.weight")
            elif "classifier.lin2.bias" in new_k:
                new_k = new_k.replace("module.head.head_0.classifier.lin2.bias", "module.fc.fc2.bias")
                new_k_eval = new_k_eval.replace("module.head.head_0.classifier.lin2.bias", "fc.fc2.bias")

        if "domain" in new_k:
            new_k = new_k.replace("module.domain_head.", "")

            domain_dict[new_k] = v
        else:
            train_dict[new_k] = v
            eval_dict[new_k_eval] = v
    return eval_dict  # , domain_dict


def evaluation_cascade(args):
    ##init
    acc_style = -1
    nmi_style = -1
    ari_style = -1
    checkpoint_path = os.path.join(args.load_path)
    if not os.path.exists(checkpoint_path):
        return None, None, None, None, None, None
    try:
        checkpoint = torch.load(checkpoint_path)
        if checkpoint_path.endswith('checkpoint_last.pth.tar'):
            checkpoint = checkpoint['state_dict']
            new_ckpt = checkpoint.copy()
            for k,v in new_ckpt.items():
                if "head_" in k and not "head_0" in k:
                    del checkpoint[k]
    except:
        return None, None, None, None, None, None
    load_model = remove_feature_hirerchy(
        checkpoint)  # ['train_model'] if args.use_train_model else checkpoint['eval_model']

    _net_builder = net_builder(args.net,
                               args.net_from_name, {})

    net = _net_builder(num_classes=args.num_classes)
    net.fc = FFN(encoder_q_out_dim=args.num_classes)
    net.load_state_dict(load_model)

    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    _eval_dset = SSL_Dataset(name=args.dataset, train=False, data_dir=args.data_dir, label_file=None, all=args.all,
                             unlabeled=False, cfg=args)
    eval_dset = _eval_dset.get_dset()
    eval_loader = get_data_loader(eval_dset,
                                  args.batch_size,
                                  num_workers=1)
    labels_pred = []
    labels_gt = []
    scores = []
    embs = []
    with torch.no_grad():
        for image, target, _, _, _ in eval_loader:
            image = image.type(torch.FloatTensor).cuda()
            logit, emb = net(image)
            scores.append(torch.softmax(logit, -1))
            labels_pred.append(torch.max(logit, dim=-1)[1].cpu().numpy())
            labels_gt.append(target.cpu().numpy())
            embs.append(emb)
    labels_pred = np.concatenate(labels_pred, axis=0)
    labels_gt = np.concatenate(labels_gt, axis=0)
    scores = torch.cat(scores, dim=0)
    embs = torch.cat(embs, dim=0)
    acc = calculate_acc(labels_pred, labels_gt)
    print(acc)
    if not isinstance(acc, float) and args.use_centers:
        center_path = checkpoint_path[:-25] + "centers.pth"
        if os.path.exists(center_path):
            non_chosen_center = torch.load(center_path)[acc]
            dis = torch.einsum('cd,nd->cn',
                               [non_chosen_center / torch.norm(non_chosen_center, dim=1).unsqueeze(-1), embs])
            for i, ac in enumerate(acc):
                idx_select = torch.argsort(dis[i:i + 1], dim=1, descending=True)[:, :3].flatten()
                labels_pred[idx_select.cpu()] = ac

            acc = calculate_acc(labels_pred, labels_gt)
        else:
            print('use_centers was used but center file does not exist - exiting')
            sys.exit

    nmi = calculate_nmi(labels_pred, labels_gt)
    ari = calculate_ari(labels_pred, labels_gt)

    return acc, nmi, ari, acc_style, nmi_style, ari_style


def evaluate_on_dir(args):
    if "spice_self" in args.dir2run_all:
        query = "/checkpoint_select.pth.tar"
        args.load_path = args.dir2run_all + "/" + query

        if not os.path.exists(args.load_path):
            print('Using checkpoint last as select is not available')
            query = "/checkpoint_last.pth.tar"
            args.load_path = args.dir2run_all + "/" + query

    acc, nmi, ari, acc_style, nmi_style, ari_style = evaluation_cascade(args)
    if acc is not None:
        print("#####################################################################################")
        print("#####################################################################################")
        print("Evaluating checkpoint:",
              args.load_path)  # .replace("//","/").split("/")[-3],args.load_path.replace("//","/").split("/")[-1])
        print("#####################################################################################")
        print("#####################################################################################")
        print(f"Test Accuracy: {acc}, NMI: {nmi}, ARI: {ari}")
        print(f"Test Accuracy Style: {acc_style}, NMI: {nmi_style}, ARI: {ari_style}")


if __name__ == "__main__":
    ###############################################################
    ######### Only argument need changing is dir_and_regex  #######
    ######### directory need to be changed by dataset name  #######
    ######### in the last hierarchy change regex between ** #######
    ######### Note: office31 expect one:_ rest expct two _  #######
    ###############################################################
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_and_regex', type=str, default='./spice/results/pacs/**')
    parser.add_argument('--dir2run_all', type=str, default='')
    parser.add_argument('--base_folder', type=str, default='./')
    parser.add_argument('--load_path', type=str,
                        default='./spice/results/pacs/cartoon_photo_sketch/spice_semi/model_0.pth')
    parser.add_argument('--data_dir', type=str, default='/datasets/')
    parser.add_argument('--domain_names', type=str, default='artpainting')
    parser.add_argument('--dataset', type=str, default='office31')
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--net', type=str, default='resnet18')

    parser.add_argument('--scores_path', type=str, default=None)
    parser.add_argument('--use_train_model', action='store_true')
    parser.add_argument('--use_centers', action='store_true')
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)
    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--label_file', type=str, default=None)
    parser.add_argument('--trans1', type=str, default="office31_none")
    parser.add_argument('--trans2', type=str, default="office31_none")
    parser.add_argument('--all', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--unlabeled', type=bool, default=False)
    parser.add_argument('--domains2comparewith', default="", type=str)
    args = parser.parse_args()

    domains_per_dataset = {"pacs": ["cartoon", "photo", "artpainting", "sketch"],
                           "officehome": ["RealWorld", "Clipart", "Product", "Art"],
                           "office31": ["amazon", "dslr", "webcam"],
                           "domainnet": ["clipart", "infograph", "quickdraw", "painting", "real", "sketch"],
                           "domainnet_small": ["clipart", "infograph", "quickdraw", "painting", "real", "sketch"]}
    if args.dir_and_regex != "":
        args.dataset = args.dir_and_regex.split("/")[-2]
        if args.dataset not in domains_per_dataset:
            print("Dataset not available!!")
            sys.exit()

        args.base_folder = '/'.join(map(str, args.dir_and_regex.split("/")[:-3])) + '/'
        args.data_dir = args.base_folder + args.data_dir + args.dataset
    else:
        args.data_dir = os.path.join(args.data_dir, args.dataset)

    args.type = args.dataset

    args.root_folder = args.data_dir

    if args.dir_and_regex != "":
        domainstrainedon = args.dir_and_regex.split("*")[-2].split("_")
        if len(domainstrainedon) == 2 and not args.dataset == "pacs" or len(domainstrainedon) == 3:
            for d in domains_per_dataset[args.dataset]:
                if d not in domainstrainedon:
                    args.domain_names = d
                    break

    if args.dataset == "office31":
        args.num_classes = 31
    elif args.dataset == "pacs":
        args.num_classes = 7
    elif args.dataset == "officehome":
        args.num_classes = 65
    elif args.dataset == "domainnet":
        args.num_classes = 345
    elif args.dataset == "domainnet_small":
        args.num_classes = 20
    else:
        raise NotImplementedError

    args.num_cluster = args.num_classes
    print("Evaluating on:", args.domain_names)

    if args.dir_and_regex != "":
        dir_name = "plots/" + args.dir_and_regex.split("/")[-1].replace("*", "___")
        runs = glob.glob(args.dir_and_regex)
        # just for nicer printing
        print("Runs found:")
        _ = [print(run.split("/")[-1]) for run in runs]
        for run in runs:
            args.dir2run_all = run + "/spice_self/"
            evaluate_on_dir(args)
