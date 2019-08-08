# coding: utf-8


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_sentence_to_token(sentence, seq_length, tokenizer):
    tokenized_text = tokenizer.tokenize(sentence)

    if len(tokenized_text) > seq_length - 2:
        tokenized_text = tokenized_text[0:(seq_length - 2)]

    position = []
    special = []
    isSpecial = False

    whole_word = ''
    words = []

    start_pos = len(tokenized_text) + 2

    connect_sign = 0

    for index in range(len(tokenized_text) - 1):

        if tokenized_text[index + 1] == "-" and tokenized_text[index + 2] != "-":
            whole_word += tokenized_text[index]
            special.append(start_pos + index)
            continue

        if tokenized_text[index] == "-":

            whole_word += tokenized_text[index]
            special.append(start_pos + index)

            if tokenized_text[index - 1] == "-":
                words.append(whole_word)
                position.append(special)
                special = []
                whole_word = ''
            continue

        if tokenized_text[index] != "-" and tokenized_text[index - 1] == "-":
            whole_word += tokenized_text[index]
            words.append(whole_word)
            special.append(start_pos + index)
            whole_word = ''
            position.append(special)
            special = []
            continue

        if tokenized_text[index + 1][0:2] == "##":
            special.append(start_pos + index)
            whole_word += tokenized_text[index]
            isSpecial = True
            continue
        else:
            if isSpecial:
                isSpecial = False
                special.append(start_pos + index)
                position.append(special)
                whole_word += tokenized_text[index]
                whole_word = whole_word.replace('##', '')
                words.append(whole_word)
                whole_word = ''
                special = []
            else:
                position.append(start_pos + index)
                words.append(tokenized_text[index])

    if isSpecial:
        isSpecial = False
        special.append(start_pos + index + 1)
        position.append(special)
        whole_word += tokenized_text[index + 1]
        whole_word = whole_word.replace('##', '')
        words.append(whole_word)
    else:
        position.append(start_pos + index + 1)
        words.append(tokenized_text[index + 1])

    return tokenized_text, words, position


def convert_whole_word_to_feature(tokens_a, mask_position, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    # tokens_a = tokenizer.tokenize(sentence)
    # print(mask_position)

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)

    tokens.append("[SEP]")
    input_type_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(1)

    tokens.append("[SEP]")
    input_type_ids.append(1)

    true_word = ''
    ind = 0
    for pos in mask_position:
        true_word = true_word + tokens[pos]
        if (ind == 0):
            tokens[pos] = '[MASK]'
        else:
            del tokens[pos]
            del input_type_ids[pos]
        ind = ind + 1

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return InputFeatures(unique_id=0, tokens=tokens, input_ids=input_ids, input_mask=input_mask,
                         input_type_ids=input_type_ids)


def convert_token_to_feature(tokens_a, mask_position, seq_length, tokenizer):
    """
        Loads a data file into a list of `InputFeature`s.
    """

    tokens = []
    input_type_ids = []

    tokens.append("[CLS]")
    input_type_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)

    tokens.append("[SEP]")
    input_type_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(1)

    tokens.append("[SEP]")
    input_type_ids.append(1)

    true_word = ''

    if isinstance(mask_position, list):
        for pos in mask_position:
            true_word = true_word + tokens[pos]
            tokens[pos] = '[MASK]'
    else:
        true_word = tokens[mask_position]
        tokens[mask_position] = '[MASK]'

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return InputFeatures(unique_id=0, tokens=tokens, input_ids=input_ids, input_mask=input_mask,
                         input_type_ids=input_type_ids)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
