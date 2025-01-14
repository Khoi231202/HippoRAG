import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from langchain_util import LangChainModel
from qa.qa_reader import qa_read
from hipporag import HippoRAG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--query', type=str)
    args = parser.parse_args()

    hipporag = HippoRAG(corpus_name=args.dataset, extraction_model='together', extraction_model_name='meta-llama/Llama-3-8b-chat-hf',
                 graph_creating_retriever_name='facebook/contriever', qa_model=LangChainModel('together', 'meta-llama/Llama-3-8b-chat-hf'))

    qa_few_shot_samples = None
    queries = [args.query]
    for query in queries:
        ranks, scores, logs = hipporag.rank_docs(query, top_k=2)
        retrieved_passages = [hipporag.get_passage_by_idx(rank) for rank in ranks]
        response = qa_read(query, retrieved_passages, qa_few_shot_samples, hipporag.qa_model)
        if not logs or any(score[2] < 0.8 for score in logs['linked_node_scores']):
          response = 'Rất xin lỗi! Tôi không thể trả lời câu hỏi của bạn.'
        print(f"{response=}")
        print(ranks)
        print(scores)
        print(logs)