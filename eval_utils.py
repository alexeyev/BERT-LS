

def eval_ss_scores(ss, labels):
    assert len(ss) == len(labels)

    potential = 0
    instances = len(ss)
    precision = 0
    precision_all = 0
    recall = 0
    recall_all = 0

    for i in range(len(ss)):

        one_prec = 0
        common = list(set(ss[i]).intersection(labels[i]))

        if len(common) >= 1:
            potential += 1

        precision += len(common)
        recall += len(common)
        precision_all += len(ss[i])
        recall_all += len(labels[i])

    potential /= instances
    precision /= precision_all
    recall /= recall_all
    F_score = 2 * precision * recall / (precision + recall)

    return potential, precision, recall, F_score


def evaluation_pipeline_scores(substitution_words, source_words, gold_words):
    instances = len(substitution_words)
    precision = 0
    accuracy = 0
    changed_proportion = 0

    for sub, source, gold in zip(substitution_words, source_words, gold_words):

        if sub == source or (sub in gold):
            precision += 1
        if sub != source and (sub in gold):
            accuracy += 1
        if sub != source:
            changed_proportion += 1

    return precision / instances, accuracy / instances, changed_proportion / instances


