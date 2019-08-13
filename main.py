#!/usr/bin/python
# -*- coding: UTF-8 -*-
import argparse
import logging
import os
import random

import numpy as np
import torch
from nltk.stem import PorterStemmer
from pytorch_transformers import BertTokenizer
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.modeling_bert import BertForMaskedLM

from eval_utils import eval_ss_scores, evaluation_pipeline_scores
from preprocess_utils import convert_whole_word_to_feature, convert_token_to_feature, convert_sentence_to_token
from read_utils import get_word_map, get_word_count, read_eval_dataset, read_eval_index_dataset
from selection import substitution_selection, lm_score, raw_score_substitutions, substitution_ranking

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--eval_path", default="eval_data/BenchLS.txt", type=str,  # required=True,
                    help="The evaluation data path, includes file name as well!")
parser.add_argument("--bert_model", default="bert-large-uncased-whole-word-masking", type=str,  # required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                         "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--output_SR_file", default="output/output.txt", type=str,  # required=True,
                    help="The output directory of writing substitution selection.")
parser.add_argument("--word_embeddings", default="crawl-300d-2M-subword.vec", type=str,
                    help="The path to the word embeddings file")
parser.add_argument("--word_frequency", default="frequency_merge_wiki_child.txt", type=str,
                    help="The path to the word frequency file")

# Other parameters
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")

parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")

parser.add_argument("--do_eval", default=True, action='store_true', help="Whether to run eval on the dev set.")
parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
parser.add_argument("--num_selections", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--num_eval_epochs", default=1, type=int, help="Total number of training epochs to perform.")

parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
# parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
parser.add_argument('--fp16', default=True, action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")


# parser.add_argument('--loss_scale', type=float, default=0,
#                     help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
#                          "0 (default value): dynamic loss scaling.\n"
#                          "Positive power of 2: static loss scaling value.\n")
# parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
# parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def extract_context(words, mask_index, window):
    """ Extracts 7 words around the content word """
    total_length = len(words)
    half_window = int(window / 2)
    assert 0 <= mask_index < total_length
    return words[max(0, mask_index - half_window):min(total_length, mask_index + half_window + 1)]


def main():
    args = parser.parse_args()
    DEVICE = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print('Using device:', DEVICE)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_eval:
        raise ValueError("At least `do_eval` must be True.")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # reading the pretrained model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))

    pretrained_bert_model = BertForMaskedLM.from_pretrained(args.bert_model, cache_dir=cache_dir)

    if args.fp16:
        pretrained_bert_model.half()
    pretrained_bert_model.to(DEVICE)

    print("Loading embeddings ...")

    word_vec_path = args.word_embeddings
    wv_dict, wv_emb = get_word_map(word_vec_path)

    print("Loaded. Loading word counts...")

    word_count_path = args.word_frequency
    word_count = get_word_count(word_count_path)

    print("Loaded.")

    stemmer_for_matching = PorterStemmer()

    SS = []
    substitution_words = []
    source_words = []

    window_context = 11

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        file_name = args.eval_path.split('/')[-1][:-4]

        if file_name == 'lex.mturk':
            eval_examples, mask_words, mask_labels = read_eval_dataset(args.eval_path)
        else:
            eval_examples, mask_words, mask_labels = read_eval_index_dataset(args.eval_path)

        eval_size = len(eval_examples)

        logger.info("Running evaluation")
        logger.info("Num examples = %d", eval_size)

        # disable training mode for BERT
        pretrained_bert_model.eval()

        for i in range(eval_size):

            source_word = mask_words[i]

            print("Sentence {} rankings: ".format(i))
            tokens, words, position = convert_sentence_to_token(eval_examples[i], args.max_seq_length, tokenizer)

            assert len(words) == len(position)

            mask_index = words.index(source_word)
            mask_context = extract_context(words, mask_index, window_context)
            mask_position = position[mask_index]

            if isinstance(mask_position, list):
                feature = convert_whole_word_to_feature(tokens, mask_position, args.max_seq_length, tokenizer)
            else:
                feature = convert_token_to_feature(tokens, mask_position, args.max_seq_length, tokenizer)

            tokens_tensor = torch.tensor([feature.input_ids]).to(DEVICE)
            token_type_ids = torch.tensor([feature.input_type_ids]).to(DEVICE)
            attention_mask = torch.tensor([feature.input_mask]).to(DEVICE)

            # Predict all tokens
            with torch.no_grad():
                prediction_scores = pretrained_bert_model(tokens_tensor, token_type_ids, attention_mask)
                if isinstance(prediction_scores, tuple):
                    prediction_scores = prediction_scores[0]

            if isinstance(mask_position, list):
                predicted_top = prediction_scores[0, mask_position[0]].topk(20)
            else:
                predicted_top = prediction_scores[0, mask_position].topk(20)

            pre_tokens = tokenizer.convert_ids_to_tokens(predicted_top[1].cpu().numpy())
            initial_subs_pool = substitution_selection(source_word, pre_tokens, stemmer_for_matching, args.num_selections)

            print("\n\n")
            print(initial_subs_pool)

            SS.append(initial_subs_pool)

            source_words.append(source_word)

            pre_word = substitution_ranking(source_word, mask_context, initial_subs_pool, wv_dict, wv_emb, word_count,
                                            tokenizer, pretrained_bert_model, mask_labels[i], DEVICE)

            substitution_words.append(pre_word)

        with open(args.output_SR_file, "a+") as output_sr_file:

            potential, precision, recall, F_score = eval_ss_scores(SS, mask_labels)
            print("The score of evaluation for substitution selection")

            output_sr_file.write(
                "\t".join([str(item) for item in [args.num_selections, precision, recall, F_score]]) + "\t")
            print(potential, precision, recall, F_score)

            precision, accuracy, changed_proportion = \
                evaluation_pipeline_scores(substitution_words, source_words, mask_labels)
            print("The score of evaluation for full LS pipeline")
            print(precision, accuracy, changed_proportion)

            output_sr_file.write(str(precision) + '\t' + str(accuracy) + '\n')


if __name__ == "__main__":
    main()
