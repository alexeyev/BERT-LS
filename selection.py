#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch as th
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cosine


def substitution_selection(source_word, pre_tokens, stemmer, num_selection=10):
    """
        Selecting possible substitutions using a heuristic
    :param source_word: the token to be replaced
    :param pre_tokens: substitution candidates
    :param stemmer: oldschool stemmer to replace all words with the same root
    :param num_selection: how many substitutions MAX to yield
    :return:
    """

    cur_tokens = []
    source_stem = stemmer.stem(source_word)

    assert num_selection <= len(pre_tokens)

    for token in pre_tokens:

        # removing 'non-words' (BPE suffixes)
        if token[0:2] == "##":
            continue

        # skipping the same word
        if token == source_word:
            continue

        token_stem = stemmer.stem(token)

        # skipping words of the same stem
        if token_stem == source_stem:
            continue

        # heuristic: skipping the long stems that begin with the same 3 letters
        if len(token_stem) >= 3 and token_stem[:3] == source_stem[:3]:
            continue

        cur_tokens.append(token)

        if len(cur_tokens) == num_selection:
            break

    if len(cur_tokens) == 0:
        cur_tokens = pre_tokens[0:num_selection + 1]

    assert len(cur_tokens) > 0

    return cur_tokens


def get_score(sentence, tokenizer, masked_lm, device):
    """
        Replacing every token with a MASK and computing masked LM scores
    """
    tokenized_input = tokenizer.tokenize(sentence)
    full_input = th.tensor([tokenizer.convert_tokens_to_ids(tokenized_input)]).to(device)
    sentence_loss = 0

    for i, word in enumerate(tokenized_input):

        # replacing word #i with a mask
        original_word = tokenized_input[i]
        tokenized_input[i] = "[MASK]"

        mask_input = th.tensor([tokenizer.convert_tokens_to_ids(tokenized_input)]).to(device)

        with th.no_grad():
            word_loss = masked_lm(mask_input, masked_lm_labels=full_input)

            if isinstance(word_loss, tuple):
                word_loss = word_loss[0]
            word_loss = word_loss.data.cpu().numpy()

        # saving loss
        sentence_loss += word_loss

        # returning the word to it's place
        tokenized_input[i] = original_word

    # exponent of the mean word-loss
    return np.exp(sentence_loss / len(tokenized_input))


def lm_score(source_word, source_context, substitutions, tokenizer, masked_lm, device):
    """
        Given a word, a contexts and possible substitutions -- scores them with the masked LM
    """

    source_sentence = " ".join(source_context)
    lm_scores = []

    for substibution in substitutions:
        # replacing the word with the possible substitution
        sub_sentence = source_sentence.replace(source_word, substibution)

        # evaluating the score with the language model
        score = get_score(sub_sentence, tokenizer, masked_lm, device)

        lm_scores.append(score)

    return lm_scores


def raw_score_substitutions(source_word, substitution_selection, wv_dict, wv_emb, word_count):
    """
        Scoring substitutions according to cosine similarity of word vectors to the replaced word's one
        and according to the counts
    """

    filtered_substitutions = []
    cosine_distance_scores = []
    count_scores = []

    is_fast = True
    source_emb = None

    # computing source word's word vector
    if source_word not in wv_dict:
        is_fast = False
        print("NOT FAST!")
    else:
        source_emb = wv_emb[wv_dict.index(source_word)].reshape(1, -1)

    for sub in substitution_selection:

        # skipping substitution not in word stats dictionary
        if sub in word_count:
            sub_count = word_count[sub]

            if is_fast:
                if sub not in wv_dict:
                    continue

                # computing substitution's word vector's distance to the source word's
                sub_embedding = wv_emb[wv_dict.index(sub)].reshape(1, -1)
                cosine_distance_scores.append(cosine(source_emb, sub_embedding))

            filtered_substitutions.append(sub)
            count_scores.append(sub_count)

    return filtered_substitutions, cosine_distance_scores, count_scores


def substitution_ranking(source_word, source_context, substitution_selection,
                         wv_dict, wv_emb, word_count, tokenizer, masked_lm, lables, device):
    filtered_subs, cosine_scores, count_scores = raw_score_substitutions(source_word, substitution_selection,
                                                                         wv_dict, wv_emb, word_count)

    cosine_scores = [item.flatten()[0] for item in cosine_scores]

    # found no replacements
    if len(filtered_subs) == 0:
        return source_word

    # computing CD-based ranks
    cosine_rank = [0] * len(cosine_scores)
    if len(cosine_scores) > 0:
        for idx, i in enumerate(np.argsort(- np.array(cosine_scores))):
            cosine_rank[i] = idx + 1

    # computing count-based ranking
    rank_count = [0] * len(count_scores)
    for idx, i in enumerate(np.argsort(- np.array(count_scores))):
        rank_count[i] = idx + 1

    # ranking with LM
    lm_scores = lm_score(source_word, source_context, filtered_subs, tokenizer, masked_lm, device)
    rank_lm = [0] * len(lm_scores)

    for idx, i in enumerate(np.argsort(np.array(lm_scores))):
        rank_lm[i] = idx + 1

    bert_rank = [i + 1 for i in range(len(filtered_subs))]
    all_ranks = [b + cd + c + lm for b, cd, c, lm in zip(bert_rank, cosine_rank, rank_count, rank_lm)]

    pre_index = np.argmin(all_ranks)
    pre_word = filtered_subs[pre_index]

    print("%s -> %s, id: %s" % (source_word, pre_word, str(pre_index)), all_ranks)

    return pre_word

# def raw_context_sis_cosine_score(source_word, sis_context, substitution_selection, wv_dict, wv_emb):
#     """
#         For each possible substitution we compute the mean cosine between the context word emb. and the substitute
#     """
#
#     context_sis = []
#     word_context = [c for c in sis_context if not c == source_word and c in wv_dict]
#
#     if len(word_context) != 0:
#
#         for sub in substitution_selection:
#
#             sub_emb = wv_emb[wv_dict.index(sub)].reshape(1, -1)
#             all_sis = 0
#
#             for context in word_context:
#                 context_word_emb = wv_emb[wv_dict.index(context)].reshape(1, -1)
#                 all_sis += cosine(sub_emb, context_word_emb)
#
#             context_sis.append(all_sis / len(word_context))
#     else:
#         context_sis = [len(substitution_selection) - i for i in range(len(substitution_selection))]
#
#     return context_sis