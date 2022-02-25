import pandas as pd
import argparse

from bert_score import BERTScorer

parser = argparse.ArgumentParser(description='Unlabeled')

parser.add_argument('--data_path', type=str, default='./tmp/tst-summarization-baseline-predict/test_generations.txt',
                    help='path to data files')

parser.add_argument('--output_data_path', type=str, default='./data/ulbl_predict.csv',
                    help='path to data files')

parser.add_argument('--reference_data_path', type=str, default='./data/ulbl_raw.csv',
                    help='path to data files')

parser.add_argument('--thres', type=float, default=0.25,
                    help='path to data files')


args = parser.parse_args()


if args.thres >= 1:
    args.thres = min(0.5, (5 - args.thres) / 6)

def main():
    hyp_path = args.data_path
    hypothesis = []
    with open(hyp_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            hypothesis.append(l[:-1])

    ref_path = args.reference_data_path
    reference = []
    test_data = pd.read_csv(ref_path)
    conv_set = [u for u in test_data['text']]
    conv = []
    for i in range(0, len(hypothesis)):
        conv.append(conv_set[i])

    
    
    if args.thres < 0:
        filtered_hypothesis = hypothesis
        filtered_conv = conv
    else:
        
        scorer = BERTScorer(lang="en", rescale_with_baseline=True)

        P, R, F1 = scorer.score(hypothesis, conv, verbose = True)

        #filtered_hypothesis = hypothesis
        #filtered_conv = conv

        filtered_hypothesis = []
        filtered_conv = []

        for i in range(0, len(hypothesis)):
            if F1[i] > args.thres:
            #if F1[i] > 0.0:
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