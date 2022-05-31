import json
import sys
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from text2vec import Similarity
from tqdm import tqdm
import sys
import jieba
sys.path.append('./Distinct-N/bin')
from metrics import distinct_n_sentence_level

class Eval:
    def __init__(self,path):
        self.in_path = path
        self.eval_data = {'data':[]}
        self.read_data()
        #句间相似度
        self.sim = Similarity(embedding_type='bert')

    def read_data(self):
        with open(self.in_path,'r',encoding='utf-8') as fp:
            lines = fp.readlines()
        print(len(lines))
        for line in lines[:]:
            if len(line.strip().split('\t'))<2:
                sent1, sent2 = line.strip().split('\t')[0],' '
            else:
                sent1,sent2 = line.strip().split('\t')
            self.eval_data['data'].append({
                'ref':sent1,
                'hyp':sent2
            })

    def text2vec_sim(self):
        total_s = 0
        for per in tqdm(self.eval_data['data'],postfix='Similarity'):
            sent1,sent2 = per['ref'],per['hyp']
            s = self.sim.get_score(sent1, sent2)
            total_s += s
            per['sim_score'] = s
        aver_score = total_s / len(self.eval_data['data'])
        self.eval_data['sim_score'] = aver_score
        return aver_score

    def distinct_2(self):
        # hypothesis = [' '.join(list(per['hyp'])) for per in self.eval_data['data']]
        hypothesis = [' '.join(jieba.cut(per['hyp'])) for per in self.eval_data['data']]
        n = 2
        print('distinct_2......')
        scores = [distinct_n_sentence_level(s, n) for s in hypothesis]
        aver_score = sum(scores) / len(scores)
        for per,score in zip(self.eval_data['data'],scores):
            per['distinct_2'] = score
        self.eval_data['distinct_2'] = aver_score
        return aver_score

    def self_bleu(self):
        ngram = 3
        weight = tuple((1. / ngram for _ in range(ngram)))
        scores = []
        for per in tqdm(self.eval_data['data'],postfix='self bleu'):
            sent1,sent2 = per['ref'],per['hyp']
            # reference, hypothesis = [list(jieba.cut(sent1))],list(jieba.cut(sent2))
            reference, hypothesis = [sent1],sent2
            score = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                    smoothing_function=SmoothingFunction().method1)
            per['inverse_self_bleu'] = 1-score
            scores.append(per['inverse_self_bleu'])
        self.eval_data['inverse_self_bleu'] = sum(scores) / len(scores)
        return sum(scores) / len(scores)

    def write_result(self):
        print('path:',self.in_path)
        print('sim_score:{}, distinct_2:{}, inverse_self_bleu:{}'.format(self.eval_data['sim_score'],
                                                                         self.eval_data['distinct_2'],
                                                                      self.eval_data['inverse_self_bleu']))
        w_path = self.in_path.replace('.txt','_eval_result.json')
        with open(w_path,'w',encoding='utf-8') as fp:
            json.dump(self.eval_data,fp,indent=2,ensure_ascii=False)

if __name__ == '__main__':
    path = sys.argv[1]
    #'../bert2gpt2_chinese/github_filt_train_10w/out.txt'
    eval = Eval(path)
    eval.text2vec_sim()
    eval.distinct_2()
    eval.self_bleu()
    eval.write_result()