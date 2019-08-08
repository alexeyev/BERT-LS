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
from sklearn.metrics.pairwise import cosine_similarity as cosine

from eval_utils import eval_ss_scores, evaluation_pipeline_scores
from preprocess_utils import convert_whole_word_to_feature, convert_token_to_feature, convert_sentence_to_token
from read_utils import get_word_map, get_word_count, read_eval_dataset, read_eval_index_dataset
from selection import substitution_selection, lm_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)


def preprocess_SR(source_word, substitution_selection, fasttext_dico, fasttext_emb, word_count):
    ss = []
    sis_scores = []
    count_scores = []
    is_fast = True
    source_emb = None

    if source_word not in fasttext_dico:
        is_fast = False
    else:
        source_emb = fasttext_emb[fasttext_dico.index(source_word)].reshape(1, -1)

    for sub in substitution_selection:

        if sub not in word_count:
            continue
        else:
            sub_count = word_count[sub]

            # if sub_count<source_count:
            #   continue
        if is_fast:
            if sub not in fasttext_dico:
                continue

            token_index_fast = fasttext_dico.index(sub)
            sis = cosine(source_emb, fasttext_emb[token_index_fast].reshape(1, -1))
            sis_scores.append(sis)

        ss.append(sub)
        count_scores.append(sub_count)

    return ss, sis_scores, count_scores


def compute_context_sis_score(source_word, sis_context, substitution_selection, fasttext_dico, fasttext_emb):
    context_sis = []
    word_context = []

    for con in sis_context:

        if con == source_word or (con not in fasttext_dico):
            continue

        word_context.append(con)

    if len(word_context) != 0:
        for sub in substitution_selection:
            sub_emb = fasttext_emb[fasttext_dico.index(sub)].reshape(1, -1)
            all_sis = 0

            for con in word_context:
                token_index_fast = fasttext_dico.index(con)
                all_sis += cosine(sub_emb, fasttext_emb[token_index_fast].reshape(1, -1))

            context_sis.append(all_sis / len(word_context))
    else:
        for i in range(len(substitution_selection)):
            context_sis.append(len(substitution_selection) - i)

    return context_sis


def substitution_ranking(source_word, source_context, substitution_selection, fasttext_dico, fasttext_emb, word_count,
                         tokenizer, masked_lm, lables):
    ss, sis_scores, count_scores = preprocess_SR(source_word, substitution_selection,
                                                 fasttext_dico, fasttext_emb, word_count)

    # print(ss)
    if len(ss) == 0:
        return source_word

    if len(sis_scores) > 0:
        seq = sorted(sis_scores, reverse=True)
        sis_rank = [seq.index(v) + 1 for v in sis_scores]

    rank_count = sorted(count_scores, reverse=True)
    count_rank = [rank_count.index(v) + 1 for v in count_scores]

    lm_scores = lm_score(source_word, source_context, ss, tokenizer, masked_lm, DEVICE)
    rank_lm = sorted(lm_scores)
    lm_rank = [rank_lm.index(v) + 1 for v in lm_scores]
    bert_rank = []

    for i in range(len(ss)):
        bert_rank.append(i + 1)

    if len(sis_scores) > 0:
        all_ranks = [bert + sis + count + LM for bert, sis, count, LM in zip(bert_rank, sis_rank, count_rank, lm_rank)]
    else:
        all_ranks = [bert + count + LM for bert, count, LM in zip(bert_rank, count_rank, lm_rank)]
    # all_ranks = [con for con in zip(context_rank)]

    pre_index = all_ranks.index(min(all_ranks))
    pre_word = ss[pre_index]

    return pre_word


def extract_context(words, mask_index, window):
    # extract 7 words around the content word

    length = len(words)
    half = int(window / 2)

    assert 0 <= mask_index < length

    context = ""

    if length <= window:
        context = words
    elif half <= mask_index < length - half:
        context = words[mask_index - half:mask_index + half + 1]
    elif mask_index < half:
        context = words[0:window]
    elif mask_index >= length - half:
        context = words[length - window:length]
    else:
        print("Wrong!")

    return context


def main():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--fp16', default=True, action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    # parser.add_argument('--loss_scale', type=float, default=0,
    #                     help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
    #                          "0 (default value): dynamic loss scaling.\n"
    #                          "Positive power of 2: static loss scaling value.\n")
    # parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    # parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_eval:
        raise ValueError("At least `do_eval` must be True.")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # train_examples = None
    # num_train_optimization_steps = None

    # reading the pretrained model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))

    pretrained_bert_model = BertForMaskedLM.from_pretrained(args.bert_model, cache_dir=cache_dir)

    if args.fp16:
        pretrained_bert_model.half()
    pretrained_bert_model.to(DEVICE)

    output_sr_file = open(args.output_SR_file, "a+")

    print("Loading embeddings ...")

    word_vec_path = args.word_embeddings
    fasttext_dico, fasttext_emb = get_word_map(word_vec_path)

    word_count_path = args.word_frequency
    word_count = get_word_count(word_count_path)

    stemmer_for_matching = PorterStemmer()

    SS = []
    substitution_words = []
    source_words = []

    bre_i = 0

    window_context = 11

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        fileName = args.eval_path.split('/')[-1][:-4]

        if fileName == 'lex.mturk':
            eval_examples, mask_words, mask_labels = read_eval_dataset(args.eval_path)
        else:
            eval_examples, mask_words, mask_labels = read_eval_index_dataset(args.eval_path)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        pretrained_bert_model.eval()
        eval_size = len(eval_examples)

        for i in range(eval_size):

            print('Sentence {} rankings: '.format(i))
            tokens, words, position = convert_sentence_to_token(eval_examples[i], args.max_seq_length, tokenizer)

            assert len(words) == len(position)

            mask_index = words.index(mask_words[i])
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

            print(type(prediction_scores), prediction_scores)
            print(mask_position, type(mask_position))

            if isinstance(mask_position, list):
                predicted_top = prediction_scores[0, mask_position[0]].topk(20)
            else:
                predicted_top = prediction_scores[0, mask_position].topk(20)

            pre_tokens = tokenizer.convert_ids_to_tokens(predicted_top[1].cpu().numpy())

            ss = substitution_selection(mask_words[i], pre_tokens,  # predicted_top[0].cpu().numpy(),
                                        stemmer_for_matching, args.num_selections)

            print('ssss------')
            print(ss)

            SS.append(ss)

            # print(mask_words[i], ":", ss)
            source_words.append(mask_words[i])

            # pre_word = substitution_ranking2(mask_words[i], ss, fasttext_dico, fasttext_emb,word_count)
            pre_word = substitution_ranking(mask_words[i], mask_context, ss, fasttext_dico, fasttext_emb, word_count,
                                            tokenizer, pretrained_bert_model, mask_labels[i])

            substitution_words.append(pre_word)


            # if(bre_i==5):
            #    break
            # bre_i += 1

        potential, precision, recall, F_score = eval_ss_scores(SS, mask_labels)

        print("The score of evaluation for substitution selection")

        output_sr_file.write(str(args.num_selections))
        output_sr_file.write('\t')
        output_sr_file.write(str(precision))
        output_sr_file.write('\t')
        output_sr_file.write(str(recall))
        output_sr_file.write('\t')
        output_sr_file.write(str(F_score))
        output_sr_file.write('\t')

        print(potential, precision, recall, F_score)

        precision, accuracy, changed_proportion = evaluation_pipeline_scores(substitution_words,
                                                                             source_words,
                                                                             mask_labels)
        print("The score of evaluation for full LS pipeline")
        print(precision, accuracy, changed_proportion)

        output_sr_file.write(str(precision))
        output_sr_file.write('\t')
        output_sr_file.write(str(accuracy))
        output_sr_file.write('\n')

        # output_sr_file.close()


if __name__ == "__main__":
    main()
