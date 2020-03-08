import random
import time

import numpy as np
import torch
# import shutil
import sys
sys.path.append("../../")
import models.bert.util as util
from transformers import AdamW, BertForSequenceClassification, BertTokenizer, WarmupLinearSchedule

from common.constants import *
from common.evaluators.bert_evaluator import BertEvaluator
from common.trainers.bert_trainer import BertTrainer
from datasets.bert_processors.aapd_processor import AAPDProcessor
from datasets.bert_processors.agnews_processor import AGNewsProcessor
from datasets.bert_processors.imdb_processor import IMDBProcessor
from datasets.bert_processors.reuters_processor import ReutersProcessor
from datasets.bert_processors.sogou_processor import SogouProcessor
from datasets.bert_processors.sst_processor import SST2Processor
from datasets.bert_processors.yelp2014_processor import Yelp2014Processor
from models.bert.args import get_args


def evaluate_split(model, processor, tokenizer, args, split='dev'):
    evaluator = BertEvaluator(model, processor, tokenizer, args, split)
    accuracy, precision, recall, f1, avg_loss = evaluator.get_scores(silent=True)[0]
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), accuracy, precision, recall, f1, avg_loss))


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    print('Device:', str(device).upper())
    print('Number of GPUs:', n_gpu)
    print('FP16:', args.fp16)

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    dataset_map = {
        'SST-2': SST2Processor,
        'Reuters': ReutersProcessor,
        'IMDB': IMDBProcessor,
        'AAPD': AAPDProcessor,
        'AGNews': AGNewsProcessor,
        'Yelp2014': Yelp2014Processor,
        'Sogou': SogouProcessor
    }

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu
    args.num_labels = dataset_map[args.dataset].NUM_CLASSES
    args.is_multilabel = dataset_map[args.dataset].IS_MULTILABEL
    prune_mask = args.prune_weight

    # if not args.trained_model:
    #     save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME)
    #     os.makedirs(save_path, exist_ok=True)

    if not args.trained_model:
        save_path = os.path.join('model_weights', dataset_map[args.dataset].NAME, args.save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        # else:
        #     shutil.rmtree(save_path)
        #     os.makedirs(save_path)
        # else:
        #     raise FileExistsError("Output directory exists!!!!!")

    args.is_hierarchical = False
    processor = dataset_map[args.dataset]()
    # pretrained_vocab_path = PRETRAINED_VOCAB_ARCHIVE_MAP[args.model]
    pretrained_vocab_path = args.model
    tokenizer = BertTokenizer.from_pretrained(pretrained_vocab_path)

    train_examples = None
    num_train_optimization_steps = None
    if not args.trained_model:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs

    # Pruning
    pruned_path = os.path.join(save_path, "init_pruned_network")

    # Retrain
    print("--- Retraining ---")
    pruned_model = BertForSequenceClassification.from_pretrained(pruned_path, prune_mask=prune_mask)

    if args.fp16:
        pruned_model.half()
    pruned_model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(pruned_model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install NVIDIA Apex for FP16 training")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.lr,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=0.01, correct_bias=False)
        scheduler = WarmupLinearSchedule(optimizer, t_total=num_train_optimization_steps,
                                         warmup_steps=args.warmup_proportion * num_train_optimization_steps)

    trainer = BertTrainer(pruned_model, optimizer, processor, scheduler, tokenizer, args, True, save_path)
    trainer.train()
    torch.save(pruned_model, trainer.snapshot_path)

    # Retest the accuracy
    print("--- After Retraining ---")
    evaluate_split(pruned_model, processor, tokenizer, args, split='dev')
    evaluate_split(pruned_model, processor, tokenizer, args, split='test')
    util.print_nonzeros(model)
