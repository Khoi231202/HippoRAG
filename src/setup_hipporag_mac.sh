data='vinfast'  # e.g., 'sample'
retriever_name='facebook/contriever'  # e.g., 'facebook/contriever'
extraction_model='meta-llama/Llama-3-8b-chat-hf' # e.g., 'gpt-3.5-turbo-1106' (OpenAI), 'meta-llama/Llama-3-8b-chat-hf' (Together AI)
available_gpus=0
syn_thresh=0.8 # float, e.g., 0.8
llm_api='together' # e.g., 'openai', 'together'
extraction_type=ner
num_passages=10
export TOGETHER_API_KEY=01d97607c24fd3c286be2a4e60a3bc05bad446db57d516bef7be860b07264741

# Running Open Information Extraction
python src/openie_with_retrieval_option_parallel.py --dataset $data --llm $llm_api --model_name $extraction_model --run_ner --num_passages all # NER and OpenIE for passages

# Creating Contriever Graph
python src/create_graph.py --dataset $data --model_name $retriever_name --extraction_model $extraction_model --threshold $syn_thresh --extraction_type $extraction_type --cosine_sim_edges

# Getting Nearest Neighbor Files
CUDA_VISIBLE_DEVICES=0
python src/RetrievalModule.py --retriever_name $retriever_name --string_filename output/query_to_kb.tsv
python src/RetrievalModule.py --retriever_name $retriever_name --string_filename output/kb_to_kb.tsv
python src/RetrievalModule.py --retriever_name $retriever_name --string_filename output/rel_kb_to_kb.tsv

python src/create_graph.py --dataset $data --model_name $retriever_name --extraction_model $extraction_model --threshold $syn_thresh --create_graph --extraction_type $extraction_type --cosine_sim_edges

python src/hippo_test.py --dataset $data --query "giới thiệu vingroup"