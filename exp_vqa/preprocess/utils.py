"""
Utilities for preprocessing sequence data.

Special tokens that are in all dictionaries:

<NULL>: Extra parts of the sequence that we should ignore
<START>: Goes at the start of a sequence
<END>: Goes at the end of a sequence, before <NULL> tokens
<UNK>: Out-of-vocabulary words
"""

SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}


def tokenize(s, delim=' ',
             add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.

    其实就是默认把句子按照空格分开成单独的词，在列表头加入<START>，在列表尾加入<END>
    punct_to_keep: 保留的词
    punct_to_remove: 移除的词
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    # if delim='' then regard the whole s as a token
    tokens = s.split(delim) if delim else [s]
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def build_vocab(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None, add_special=None):
    """

    """
    token_to_count = {}
    # 分词的参数
    tokenize_kwargs = {
        'delim': delim,
        'punct_to_keep': punct_to_keep,
        'punct_to_remove': punct_to_remove,
    }
    for seq in sequences:
        # 每一个句子的分词结果列表
        seq_tokens = tokenize(seq, **tokenize_kwargs,
                              add_start_token=False, add_end_token=False)
        for token in seq_tokens:
            # 如果分词的token不在计数字典的键值里，则添加这个token为键值
            if token not in token_to_count:
                token_to_count[token] = 0
            # 检测到对应的token，在指定计数位置加一
            token_to_count[token] += 1

    token_to_idx = {}
    # 这个len(token_to_idx)的作用是按顺序编号所有token：0，1，2，3...
    if add_special:
        # 遍历一开始定义的特殊字符列表字典
        for token in SPECIAL_TOKENS:
            token_to_idx[token] = len(token_to_idx)
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            # 只有计数大于等于min_token_count才计入词典
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
    """
  把句子的token列表依据词典转换为对应的idx列表
  """
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    """
  把idx列表依据逆词典转换为对应的token列表
  """
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)
