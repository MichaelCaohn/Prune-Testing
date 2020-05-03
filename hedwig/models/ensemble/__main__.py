import random
import time

import numpy as np
import torch
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
from models.ensemble.args import get_args


def evaluate_split(model, processor, tokenizer, args, split='dev'):
    evaluator = BertEvaluator(model, processor, tokenizer, args, split)
    accuracy, precision, recall, f1, avg_loss = evaluator.get_scores(silent=True)[0]
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), accuracy, precision, recall, f1, avg_loss))

def ensemble_cal(model, processor, tokenizer, args, split='dev'):
    evaluator = BertEvaluator(model, processor, tokenizer, args, split)
    label, prediction = evaluator.get_pred(silent=True)
    return label, prediction

def ensemble_acc(model, processor, tokenizer, args, final_pred, label, split='dev'):
    evaluator = BertEvaluator(model, processor, tokenizer, args, split)
    accuracy, precision, recall, f1 = evaluator.get_accuracy(final_pred, label, silent=True)
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    print("Saving path: {}".format(args.save_path))
    print("Data set: {}".format(args.dataset))
    print("Learning hyperparamters: lr:{}".format(args.lr))
    print("Learning hyperparamters: ep:{}".format(args.epochs))

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

    # if not args.trained_model:
    #     save_path = os.path.join("model_weights", dataset_map[args.dataset].NAME, args.save_path)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)

    args.is_hierarchical = False
    processor = dataset_map[args.dataset]()
    tokenizer_uncased = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_cased = BertTokenizer.from_pretrained("bert-base-cased")
    path_list = ["bert-base-uncased-lr15_ep4", "bert-base-uncased-lr175-ep3",
                 "bert-base-uncased-lr175-ep4", "bert-base-uncased-lr2_ep4", "bert-base-uncased-lr15_ep5",
                 "bert-base-uncased-lr175_ep5", "bert-base-uncased-lr2-ep5",
                 "bert-base-uncased-lr15_ep6", "bert-base-uncased_lr175_ep6",
                 "bert-large-uncased_lr175_ep3", "bert-large-uncased_lr15_ep4",
                 "bert-large-uncased_lr175_ep4", "bert-large-uncased_lr2_ep4",
                 "bert-large-uncased_lr15_ep5", "bert-large-uncased_lr175_ep5",
                 "bert-large-uncased_lr2_ep5", "bert-large-uncased_lr15_ep6",
                 "bert-large-cased-lr175_ep4", "bert-base-cased-lr15-ep4",
                 "bert-base-cased-lr2_ep4"]
    num_list = args.ensemble_members.split('+')
    print(num_list)
    final_pred = 0
    model_list = [1]
    model_list[0] = path_list[int(num_list[0]) - 1]
    for i in range(1, len(num_list)):
        print(num_list[i])
        model_list.append(path_list[int(num_list[i]) - 1])
    tot = len(model_list)
    cnt_base = 0
    cnt_large = 0
    print(model_list)
    for j in model_list:
        if 'uncased' in j:
            if 'base' in j:
                cnt_base += 1
            else:
                cnt_large += 1
    cnt_case = tot - cnt_base - cnt_large
    case_weight = 1
    if cnt_case == 0:
        base_uncased_weight = 1
    else:
        base_uncased_weight = cnt_case
    large_uncased_weight = base_uncased_weight + 0.5
    print("Number of case model: {}".format(cnt_case))
    print("Number of base-uncase model: {}".format(cnt_base))
    print("Number of large-uncase model: {}".format(cnt_large))

    for i in range(len(model_list)):
        pre_path = os.path.join(model_list[i], "pruned_model_weight.pth")
        before_path = os.path.join("model_weights", dataset_map[args.dataset].NAME)
        
        tar_path = os.path.join(before_path, pre_path)
        if "bert-base-uncased" in model_list[i]:
            args.model = "bert-base-uncased"
        elif "bert-base-cased" in model_list[i]:
            args.model = "bert-base-cased"
        elif "bert-large-cased" in model_list[i]:
            args.model = "bert-large-cased"
        else:
            args.model = "bert-large-uncased"
        model = BertForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels,
                                                              prune_mask=args.prune_weight)
        model_ = torch.load(tar_path, map_location=lambda storage, loc: storage)
        state = {}
        for key in model_.state_dict().keys():
            new_key = key.replace("module.", "")
            state[new_key] = model_.state_dict()[key]
        model.load_state_dict(state)
        model = model.to(device)

        if "uncased" in model_list[i]:
            tokenizer = tokenizer_uncased
            args.model_name_or_path = "bert-base-uncased"
        else:
            tokenizer = tokenizer_cased
            args.model_name_or_path = "bert-base-cased"

        label, pred = ensemble_cal(model, processor, tokenizer, args, split='dev')

        
        if 'uncased' in tar_path:
            if 'base' in tar_path:
                weight = base_uncased_weight
            else:
                weight = large_uncased_weight
        else:
            weight = case_weight
        print("Weight for BERT: {} with weight: {}".format(tar_path, weight))
        if i == 0:
            final_pred = weight * pred
        else:
            final_pred += weight * pred
        temp_save_path = "model_after_retraining.pth"
        torch.save(model, temp_save_path)
        
    # pred_label = np.argmax(final_pred, axis=1)
    accuracy, precision, recall, f1 = ensemble_acc(model, processor, tokenizer, args, final_pred, label, split='dev')
    # result = compute_metrics(eval_task, pred_label, out_label_ids)
    print("Ensembeled accuracy: {}, precision: {}, recall: {}, f1: {}".format(accuracy, precision, recall, f1))
    accuracy, precision, recall, f1 = ensemble_acc(model, processor, tokenizer, args, final_pred, label, split='test')
    # result = compute_metrics(eval_task, pred_label, out_label_ids)
    print("Ensembeled accuracy: {}, precision: {}, recall: {}, f1: {}".format(accuracy, precision, recall, f1))

    # pretrained_vocab_path = args.model
    # tokenizer = BertTokenizer.from_pretrained(pretrained_vocab_path)

    # train_examples = None
    # num_train_optimization_steps = None
    # if not args.trained_model:
    #     train_examples = processor.get_train_examples(args.data_dir)
    #     num_train_optimization_steps = int(
    #         len(train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs
    #
    #
    # pretrained_model_path = os.path.join(save_path, "model_weight.pth")
    # model = BertForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels,
    #                                                       prune_mask=args.prune_weight)
    # model_ = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
    # state = {}
    # for key in model_.state_dict().keys():
    #     new_key = key.replace("module.", "")
    #     state[new_key] = model_.state_dict()[key]
    # model.load_state_dict(state)
    # model = model.to(device)
    #
    # evaluate_split(model, processor, tokenizer, args, split='dev')
    # evaluate_split(model, processor, tokenizer, args, split='test')
    # # model.prune_by_std(args.sensitivity)
    # # print("--- After pruning ---")
    # # util.print_nonzeros(model)
    # # if args.fp16:
    # #     model.half()
    # # model.to(device)
    # #
    # # if n_gpu > 1:
    # #     model = torch.nn.DataParallel(model)
    # #
    # # # Prepare optimizer
    # # param_optimizer = list(model.named_parameters())
    # # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # # optimizer_grouped_parameters = [
    # #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    # #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # #
    # # if args.fp16:
    # #     try:
    # #         from apex.optimizers import FP16_Optimizer
    # #         from apex.optimizers import FusedAdam
    # #     except ImportError:
    # #         raise ImportError("Please install NVIDIA Apex for FP16 training")
    # #
    # #     optimizer = FusedAdam(optimizer_grouped_parameters,
    # #                           lr=args.lr,
    # #                           bias_correction=False,
    # #                           max_grad_norm=1.0)
    # #     if args.loss_scale == 0:
    # #         optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    # #     else:
    # #         optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    # #
    # # else:
    # #     optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=0.01, correct_bias=False)
    # #     scheduler = WarmupLinearSchedule(optimizer, t_total=num_train_optimization_steps,
    # #                                      warmup_steps=args.warmup_proportion * num_train_optimization_steps)
    #
    # # print("--- Retraining ---")
    # # trainer = BertTrainer(model, optimizer, processor, scheduler, tokenizer, args, True, save_path)
    # # prune_start_time = time.time()
    # # trainer.train()
    # # prune_elapsed_time = time.time() - prune_start_time
    # # pruned_model = torch.load(trainer.snapshot_path)
    # # print("--- After Retraining ---")
    # # evaluate_split(model, processor, tokenizer, args, split='dev')
    # # evaluate_split(model, processor, tokenizer, args, split='test')
    # # print("Total training time: {}".format(prune_elapsed_time / 60))