# coding: utf-8
import torch as th
import numpy as np


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

        # removing non-words
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


def get_score(sentence, tokenizer, masked_lm, device="cpu"):
    """
        Replacing every token with a MASK and computing masked LM scores
    """
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = th.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
    sentence_loss = 0

    for i, word in enumerate(tokenize_input):
        original_word = tokenize_input[i]
        tokenize_input[i] = "[MASK]"

        mask_input = th.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)

        with th.no_grad():
            word_loss = masked_lm(mask_input, masked_lm_labels=tensor_input)
            if isinstance(word_loss, tuple):
                word_loss = word_loss[0]
            word_loss = word_loss.data.cpu().numpy()

        sentence_loss += word_loss
        tokenize_input[i] = original_word

    return np.exp(sentence_loss / len(tokenize_input))


def lm_score(source_word, source_context, substitution_selection, tokenizer, masked_lm):

    source_sentence = " ".join(source_context)
    lm_scores = []

    for substibution in substitution_selection:
        # replacing the word with the possible substitution
        # todo: make more effective
        sub_sentence = source_sentence.replace(source_word, substibution)

        # evaluating the score with the language model
        score = get_score(sub_sentence, tokenizer, masked_lm)

        lm_scores.append(score)

    return lm_scores