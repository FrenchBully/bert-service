from bert_serving.client import BertClient
from flask import Flask
from flask import request
from flask import jsonify

import json
import numpy as np
import torch

app = Flask(__name__)
if __name__ == "__main__":
  app.run(debug=True)


@app.route('/analyze/responses', methods=['POST'])
def analyzeResponses():
    if request.is_json:
      data_dict = request.get_json()
      text = data_dict["text"]
      # ip address of the GPU machine
      bc = BertClient(ip='localhost')
      with open('./server/data/answer-corpus.csv') as readFile:
          line_count = 0
          answers = [i.strip() for i in readFile.readlines()]
      # encode corpus as array of strings
      doc_vecs = bc.encode(answers)  # if tokenized: is_tokenized=True

      while True:
          # query = input('Find matching answer: ')
          query_vec = bc.encode([text])[0]
          # convert to torch input
          tensor_query_vec = torch.from_numpy(query_vec)
          tensor_doc_vecs = torch.from_numpy(doc_vecs)
          # compute normalized dot product as score
          tensor_input = tensor_query_vec * tensor_doc_vecs
          score = torch.sum(tensor_input, 1) / \
              torch.norm(tensor_doc_vecs, dim=1)
          argsort = torch.argsort(score)
          topk_idx = torch.topk(argsort, 1)
          scores = []
          for idx in topk_idx:
              print('> %s\t%s' % (score[idx], answers[idx]))
              scores.append(answers[idx])
          print(f'Scores: {scores}')
          return jsonify(scores[-1]), 201
