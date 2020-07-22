from bert_serving.client import BertClient
import numpy as np
import csv
import torch
bc = BertClient(ip='**.**.**.**')  # ip address of the GPU machine

with open('file-to-read') as readFile:
    line_count = 0
    answers = [i.strip() for i in readFile.readlines()]
# encode corpus as array of strings
doc_vecs = bc.encode(answers)  # if tokenized: is_tokenized=True

while True:
    query = input('Find matching answer: ')
    query_vec = bc.encode([query])[0]
    # convert to torch input
    tensor_query_vec = torch.from_numpy(query_vec)
    tensor_doc_vecs = torch.from_numpy(doc_vecs)
    # compute normalized dot product as score
    tensor_input = tensor_query_vec * tensor_doc_vecs
    score = torch.sum(tensor_input, 1) / \
        torch.norm(tensor_doc_vecs, dim=1)
    argsort = torch.argsort(score)
    topk_idx = torch.topk(argsort, 1)
    nptopk = np.argsort(score)[::-1][:5]
    print(f'nptopk: {nptopk}')
    print(f'topk_idx: {topk_idx}')
    for idx in topk_idx:
        print('> %s\t%s' % (score[idx], answers[idx]))
