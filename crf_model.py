# -*- coding: utf-8 -*-
"""
CRF中文词法分析器（分词+词性标注）
使用 sklearn-crfsuite
"""

import re
import pickle
from collections import defaultdict
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
import numpy as np

def read_crf_data(filepath):
    """读取CRF格式的数据，返回字符序列和标签"""
    X = []  # 特征
    y = []  # 标签（BMES）
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sent_chars = []
    sent_labels = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if sent_chars:
                X.append(sent_chars)
                y.append(sent_labels)
                sent_chars = []
                sent_labels = []
            continue
        
        parts = line.split('\t')
        if len(parts) >= 4:
            char = parts[0]
            bmes = parts[1]
            pos = parts[2]
            char_type = parts[3]
            
            sent_chars.append({
                'char': char,
                'type': char_type,
                'bmes': bmes,
                'pos': pos
            })
            sent_labels.append(bmes)
    
    return X, y

def load_data_for_crf(filepath):
    """加载数据，返回特征和标签"""
    X = []
    y = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sent = []
    for line in lines:
        line = line.strip()
        if not line:
            if sent:
                X.append(sent2features(sent))
                y.append(sent2labels(sent))
                sent = []
            continue
        
        parts = line.split('\t')
        if len(parts) >= 4:
            sent.append({
                'char': parts[0],
                'bmes': parts[1],
                'pos': parts[2],
                'type': parts[3]
            })
    
    return X, y

def load_pos_data(filepath):
    """加载词性标注数据"""
    X = []
    y = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sent = []
    for line in lines:
        line = line.strip()
        if not line:
            if sent:
                X.append(sent2features(sent))
                y.append(sent2poslabels(sent))
                sent = []
            continue
        
        parts = line.split('\t')
        if len(parts) >= 4:
            sent.append({
                'char': parts[0],
                'bmes': parts[1],
                'pos': parts[2],
                'type': parts[3]
            })
    
    return X, y

def word2features(sent, i):
    """提取单个位置的特征"""
    word = sent[i]
    features = {
        'char': word['char'],
        'type': word['type'],
    }
    
    # 添加前后字符特征（窗口大小为2）
    if i > 0:
        features['prev_char'] = sent[i-1]['char']
        features['prev_type'] = sent[i-1]['type']
    else:
        features['prev_char'] = 'BOS'
        features['prev_type'] = 'BOS'
    
    if i > 1:
        features['prev2_char'] = sent[i-2]['char']
        features['prev2_type'] = sent[i-2]['type']
    else:
        features['prev2_char'] = 'BOS'
        features['prev2_type'] = 'BOS'
    
    if i < len(sent) - 1:
        features['next_char'] = sent[i+1]['char']
        features['next_type'] = sent[i+1]['type']
    else:
        features['next_char'] = 'EOS'
        features['next_type'] = 'EOS'
    
    if i < len(sent) - 2:
        features['next2_char'] = sent[i+2]['char']
        features['next2_type'] = sent[i+2]['type']
    else:
        features['next2_char'] = 'EOS'
        features['next2_type'] = 'EOS'
    
    # 组合特征
    if i > 0:
        features['prev_char+char'] = sent[i-1]['char'] + '_' + word['char']
    
    if i < len(sent) - 1:
        features['char+next_char'] = word['char'] + '_' + sent[i+1]['char']
    
    # 前后类型组合
    if i > 0:
        features['prev_type+type'] = sent[i-1]['type'] + '_' + word['type']
    
    return features

def sent2features(sent):
    """提取整个句子的特征"""
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    """提取BMES标签"""
    return [word['bmes'] for word in sent]

def sent2poslabels(sent):
    """提取词性标签"""
    return [word['pos'] for word in sent]

def tags_to_words(chars, tags):
    """将BMES标签转换为词序列"""
    words = []
    current_word = []
    
    for ch, tag in zip(chars, tags):
        if tag == 'B':
            if current_word:
                words.append(''.join(current_word))
            current_word = [ch]
        elif tag == 'M':
            current_word.append(ch)
        elif tag == 'E':
            current_word.append(ch)
            words.append(''.join(current_word))
            current_word = []
        else:  # 'S'
            if current_word:
                words.append(''.join(current_word))
                current_word = []
            words.append(ch)
    
    if current_word:
        words.append(''.join(current_word))
    
    return words

def evaluate_segmentation(model, X_test, y_test):
    """评测分词性能"""
    y_pred = model.predict(X_test)
    
    # 提取字符序列
    chars_list = []
    for sent in X_test:
        chars = [f['char'] for f in sent]
        chars_list.append(chars)
    
    # 词级别评测
    total_gold = 0
    total_pred = 0
    correct = 0
    
    for i, (chars, y_true, y_pred_sent) in enumerate(zip(chars_list, y_test, y_pred)):
        gold_words = tags_to_words(chars, y_true)
        pred_words = tags_to_words(chars, y_pred_sent)
        
        total_gold += len(gold_words)
        total_pred += len(pred_words)
        
        # 词级别精确匹配
        gold_set = set(gold_words)
        pred_set = set(pred_words)
        correct += len(gold_set & pred_set)
    
    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, total_gold, total_pred, correct

def predict_sentence(crf_model, sentence):
    """对单个句子进行分词"""
    chars = list(sentence)
    sent_features = []
    for i, ch in enumerate(chars):
        features = {
            'char': ch,
            'type': 'HAN' if '\u4e00' <= ch <= '\u9fff' else 'OTHER',
        }
        if i > 0:
            features['prev_char'] = chars[i-1]
        else:
            features['prev_char'] = 'BOS'
        if i > 1:
            features['prev2_char'] = chars[i-2]
        else:
            features['prev2_char'] = 'BOS'
        if i < len(chars) - 1:
            features['next_char'] = chars[i+1]
        else:
            features['next_char'] = 'EOS'
        if i < len(chars) - 2:
            features['next2_char'] = chars[i+2]
        else:
            features['next2_char'] = 'EOS'
        if i > 0:
            features['prev_char+char'] = chars[i-1] + '_' + ch
        if i < len(chars) - 1:
            features['char+next_char'] = ch + '_' + chars[i+1]
        
        sent_features.append(features)
    
    tags = crf_model.predict([sent_features])[0]
    words = tags_to_words(chars, tags)
    return words


# ==================== 主程序 ====================
print("="*60)
print("CRF中文词法分析器（分词+词性标注）")
print("="*60)

# 1. 加载数据
print("\n【步骤1】加载数据")
X_train, y_train = load_data_for_crf("crf_train.txt")
X_test, y_test = load_data_for_crf("crf_test.txt")

print(f"训练集: {len(X_train)} 个句子")
print(f"测试集: {len(X_test)} 个句子")

# 2. 训练分词模型
print("\n【步骤2】训练CRF分词模型")
crf_seg = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
    verbose=False
)

crf_seg.fit(X_train, y_train)
print("训练完成！")

# 3. 评测分词
print("\n【步骤3】分词性能评测")
print("-"*40)

precision, recall, f1, total_gold, total_pred, correct = evaluate_segmentation(
    crf_seg, X_test, y_test
)

print(f"标准词总数: {total_gold}")
print(f"预测词总数: {total_pred}")
print(f"正确匹配词数: {correct}")
print()
print(f"精确率 (Precision): {precision:.4f} ({precision*100:.2f}%)")
print(f"召回率 (Recall):    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1值:               {f1:.4f}")

# 4. 测试示例
print("\n【步骤4】测试示例")
test_sentences = [
    "我喜欢北京天安门",
    "南京市长江大桥",
    "计算机科学是人工智能的基础",
    "今天我心情有一点怪怪"
]

for sent in test_sentences:
    words = predict_sentence(crf_seg, sent)
    print(f"输入: {sent}")
    print(f"分词: {'/'.join(words)}")
    print()

# 5. 词性标注模型
print("\n【步骤5】训练CRF词性标注模型")
X_train_pos, y_train_pos = load_pos_data("crf_train.txt")
X_test_pos, y_test_pos = load_pos_data("crf_test.txt")

crf_pos = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
    verbose=False
)

crf_pos.fit(X_train_pos, y_train_pos)
print("训练完成！")

# 6. 词性标注评测
print("\n【步骤6】词性标注性能评测")
print("-"*40)

y_pred_pos = crf_pos.predict(X_test_pos)

# 展平列表计算准确率
y_true_flat = [tag for sent in y_test_pos for tag in sent]
y_pred_flat = [tag for sent in y_pred_pos for tag in sent]

correct_pos = sum(1 for t, p in zip(y_true_flat, y_pred_flat) if t == p)
total_pos = len(y_true_flat)
pos_accuracy = correct_pos / total_pos if total_pos > 0 else 0

print(f"词性标注准确率: {pos_accuracy:.4f} ({pos_accuracy*100:.2f}%)")
print(f"正确/总数: {correct_pos}/{total_pos}")

# 7. 保存模型
print("\n【步骤7】保存模型")
with open("crf_seg_model.pkl", "wb") as f:
    pickle.dump(crf_seg, f)
with open("crf_pos_model.pkl", "wb") as f:
    pickle.dump(crf_pos, f)
print("模型已保存")

print("\n" + "="*60)
print("实验完成！")
print("="*60)