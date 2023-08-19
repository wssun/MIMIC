# Mutual Information Guided Backdoor Mitigation for Pre-trained Encoders

## 1. overview

This is an implementation demo for our submission, “Mutual Information Guided Backdoor Mitigation for Pre-trained Encoders”.

In our paper, MIMIC is proposed to remove backdoors in pre-trained encoders.  We also offer a step-by-step guidance to help your build your own evaluation.

The code is tested on Ubuntu 18.04, Pytorch 1.7.1.





## 2. commands

### 2.1. step 0: Train a clean encoder

Train a clean encoder from scratch

```python
python -u train_clean_encoder.py --lr 1e-3 --results_dir <your_directory>
```

Then a clean encoder will be saved in `<your_directory>`.

### 2.2. step 1: Train a backdoored encoder

Training a backdoor model. Here, we use BadEncoder as example:

```python
 python train_badencoder.py --trigger_file trigger/trigger_pt_white_21_10_ap_replace.npz --reference_file reference/cifar10/priority.npz --pretraining_dataset cifar10 --downstreamTask gtsrb --pretrained_encoder --<your_clean_encoder> --results_dir <your_directory>
```

The  `<your_clean_encoder>` means the trained clean encoder path from step 0.

And the backdoored encoder will be saved in `<your_directory>`

### 2.3. step 2: Remove backdoors by MIMIC 

Remove Trigger using Medic, clean label attack as an example.

```python
python -u MIMIC.py --lr 1e-2 --batch_size 128 --epochs 1000 --pretraining_dataset cifar10 --teacher <your_backdoored_encoder> --ratio 0.04 --results_dir <your_directory>
```

The `<your_backdoored_encoder>` means the path to a backdoored encoder obtained in step 1.

Then a purified encoder will be saved in `<your_directory>`.

### 2.4. step 3: Train a downstream classifier

Based on the encoder purified in stage 2, we can test the ACC and ASR by training a downstream classifier:

```python
python training_downstream_classifier.py --dataset gtsrb  --reference_label 12 --trigger_file trigger/trigger_pt_white_21_10_ap_replace.npz --encoder_usage_info cifar10 --encoder <your_purified_encoder> --reference_file reference/cifar10/priority.npz --nn_epochs 1000
```

The `<your_purified_encoder>` means the path to the purified encoder obtained in step 3.

Then the results will show on the screen.





## 3.Acknowledgement

Thanks for [NAD](https://github.com/bboylyg/NAD/tree/f17b71390f61fe24335728bfea53e4fe86ee450b) and [BadEncoder](https://github.com/jinyuan-jia/BadEncoder). 

The dataset we use in based on badencoder, please download it following [link](https://github.com/jinyuan-jia/BadEncoder) in `data/`.

Their amazing implementations inspire us.

