import pandas as pd
import argparse
import pickle

import numpy as np
from bert_score import BERTScorer

parser = argparse.ArgumentParser(description='Unlabeled')

parser.add_argument('--data_path', type=str, default='./tmp/tst-summarization-baseline-predict/test_generations.txt',
                    help='path to data files')

parser.add_argument('--output_data_path', type=str, default='./data/ulbl_predict.csv',
                    help='path to data files')

parser.add_argument('--thres', type=float, default=0.0,
                    help='path to data files')


args = parser.parse_args()

def main():
    ref_path = args.output_data_path
    test_data = pd.read_csv(ref_path)
    conv = [u for u in test_data['text']]
    hypothesis = [u for u in test_data['summary']]
    
    with open(args.data_path, 'rb') as f:
        scores = pickle.load(f)
    
    filtered_hypothesis = []
    filtered_conv = []
    
    if args.thres < 0:
        index = np.argsort(scores)
        top_index = index[ int(len(scores) * args.thres) :]
    elif args.thres >= 0:
        top_index = np.where(scores > 0)[0]
        
        
    print(len(top_index))
    
    for i in top_index:
        filtered_hypothesis.append(hypothesis[i])
        filtered_conv.append(conv[i])


    data_dict = {'text':[], 'summary':[]}
    for i in range(0,len(filtered_hypothesis)):
        data_dict['text'].append(filtered_conv[i])
        data_dict['summary'].append(filtered_hypothesis[i])
    pd_data = pd.DataFrame.from_dict(data_dict)
    pd_data.to_csv(args.output_data_path, index = False)



if __name__ == "__main__":
    main()