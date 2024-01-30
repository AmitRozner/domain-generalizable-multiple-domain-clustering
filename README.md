# Domain-Generalizable Multiple-Domain Clustering
Link to our paper: https://arxiv.org/abs/2301.13530
## Installation
Please refer to [requirement.txt](./requirements.txt) for all required packages.
Tested using Python 3.8

Then, clone this repo
```shell script
git clone 
cd DomainGMDC
```
Download pre-trained model from:
[AdaIN-Style-Transfer-PyTorch](https://github.com/Maggiking/AdaIN-Style-Transfer-PyTorch/blob/master/README.md)

Put the weights in ./adain_weights folder

## Data
Prepare datasets of interest as described in [dataset.md](./dataset.md).

## Training
Example training command for OfficeHome dataset with RealWorld, Clipart, and Product domains:
```shell script
 python tools/run_end_to_end.py --domain_names Product_Clipart_RealWorld --seed 0 --embedding_batch_size 512 --domain_loss_weight 0.02 --dist-url tcp://localhost:10026 --keep_strong_heads --multi_q --balance_moco_domains --data_type officehome --num_cluster 65 --center_based_truncate --wandb_run_name <enter_run_name>  --epochs 500 --data ./datasets/ --root_save_folder ./results/ --use_wandb --arch resnet18 --soft_balance --domain_size_layers 2048 1024 512 256 128 --train_self_batch_size 256 --batch_size 8 --style_transfer --heads2keep 5 --p_bcd_augment 0.2 --self_smoothing 0.9  --pred_based_smoothing  --moco_p_bcd_augment 0.8 
 ```
The above should run both pre-training and training in one run. 

## Evaluation
An example on Officehome dataset:

```shell script
 python tools/evaluate.py --dir_and_regex '<Root_path>/spice/results/officehome/*Art_Clipart_Product*'
```
The above will find all runs in "officehome" folder which were trained using "Art" "Clipart" and "Product" domains. 
It will automatically infer the remaining domain and perform evaluation on it. 

Implemented datasets and domains are the same as in our paper: 
>"pacs": ["cartoon", "photo", "artpainting", "sketch"]
>
>"officehome": ["RealWorld", "Clipart", "Product", "Art"]
> 
>"office31": ["amazon", "dslr", "webcam"]
>
>"DomainNet": ["clipart", "infograph", "quickdraw", "painting", "real", "sketch"]

Result should include accuracy in cases it is possible to compute, otherwise it will print a list of the not 
predicted clusters.

## Acknowledgement for reference repos
- [GATCluster](https://github.com/niuchuangnn/GATCluster)
- [MOCO](https://github.com/facebookresearch/moco)
- [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification)
- [FixMatch](https://github.com/LeeDoYup/FixMatch-pytorch)
- [IIC](https://github.com/xu-ji/IIC)
- [SPICE](https://github.com/niuchuangnn/SPICE)
- [AdaIN-Style-Transfer-PyTorch](https://github.com/Maggiking/AdaIN-Style-Transfer-PyTorch/blob/master/README.md)


