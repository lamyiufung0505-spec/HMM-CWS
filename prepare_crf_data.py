# -*- coding: utf-8 -*-
"""
将人民日报语料转换为CRF训练格式
"""

import re
import os
import random

def load_tagged_corpus(filepath):
    with open(filepath, 'r', encoding='gb18030') as f:
        content = f.read()
    return content

def parse_tagged_text(text):
    pattern = r'([^/\s]+)/([a-zA-Z]+)'
    matches = re.findall(pattern, text)
    return matches

def word_to_bmes(word):
    if len(word) == 1:
        return ['S']
    elif len(word) == 2:
        return ['B', 'E']
    else:
        tags = ['B'] + ['M'] * (len(word) - 2) + ['E']
        return tags

def get_char_type(ch):
    if ch.isdigit():
        return 'NUM'
    if ch.isalpha() and ord(ch) < 128:
        return 'ENG'
    if ch in '，。！？；：""''、':
        return 'PUNC'
    if '\u4e00' <= ch <= '\u9fff':
        return 'HAN'
    return 'OTHER'

def convert_to_crf_format(input_file, output_file, max_sentences=5000):
    content = load_tagged_corpus(input_file)
    sentences = re.split(r'[。？！；\n]+', content)
    
    all_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        
        tokens = parse_tagged_text(sent)
        if len(tokens) < 1:
            continue
        
        sent_data = []
        for word, pos in tokens:
            chars = list(word)
            bmes_tags = word_to_bmes(word)
            for ch, bmes in zip(chars, bmes_tags):
                char_type = get_char_type(ch)
                sent_data.append({
                    'char': ch,
                    'bmes': bmes,
                    'pos': pos,
                    'type': char_type
                })
        
        all_sentences.append(sent_data)
        
        if len(all_sentences) >= max_sentences:
            break
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent in all_sentences:
            for item in sent:
                f.write(f"{item['char']}\t{item['bmes']}\t{item['pos']}\t{item['type']}\n")
            f.write("\n")
    
    print(f"转换完成！共 {len(all_sentences)} 个句子")
    print(f"输出文件: {output_file}")
    return all_sentences


# ==================== 主程序 ====================
print("="*50)
print("CRF数据转换")
print("="*50)

train_file = "train_pd.txt"

if not os.path.exists(train_file):
    print(f"错误：找不到 {train_file}")
    exit(1)


all_sentences = convert_to_crf_format(train_file, "crf_all.txt", max_sentences=50000)

# 分割训练集和测试集（80%训练，20%测试）
random.seed(42)
random.shuffle(all_sentences)
split_idx = int(len(all_sentences) * 0.8)

train_sents = all_sentences[:split_idx]
test_sents = all_sentences[split_idx:]

# 保存训练集
with open('crf_train.txt', 'w', encoding='utf-8') as f:
    for sent in train_sents:
        for item in sent:
            f.write(f"{item['char']}\t{item['bmes']}\t{item['pos']}\t{item['type']}\n")
        f.write("\n")

# 保存测试集
with open('crf_test.txt', 'w', encoding='utf-8') as f:
    for sent in test_sents:
        for item in sent:
            f.write(f"{item['char']}\t{item['bmes']}\t{item['pos']}\t{item['type']}\n")
        f.write("\n")

print(f"\n训练集: {len(train_sents)} 个句子")
print(f"测试集: {len(test_sents)} 个句子")
print("数据准备完成！")