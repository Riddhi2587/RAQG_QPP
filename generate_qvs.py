################################
# Generated QVs using Llama ####
# Needs to download GGUF files #
# and put them under a director#
#called gguf_storage outside ###
# working directory. ###########
# The results are NOT re-ranked#
# by RBO. ######################
################################

from llama_cpp import Llama
import pyterrier as pt
import json, json5
import pandas as pd
import sys
import os
from itertools import product
from pathlib import Path
import argparse
import re

def load_llama():
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    print(parent)
    llm = Llama(
        model_path=os.path.join(parent, 'gguf_storage', 'Meta-Llama-3-8B-Instruct.Q8_0.gguf'),
        logits_all=False,
        verbose=False,
        n_gpu_layers=-1, # Uncomment to use GPU acceleration
        n_ctx=2048, # Uncomment to increase the context window
    )
    llm.set_seed(1000)
    return llm

def llama_call(llm, prompt, temperature):
      
      output = llm(
                  prompt, # Prompt
                  max_tokens=1048, # Generate up to 1048 tokens, set to None to generate up to the end of the context window
                  stop=['</response>'], # Stop generating just before the model would generate a close tag of response
                  echo=False, # Echo the prompt back in the output
                #   logprobs=50,
                #   top_k=50,
                  temperature=temperature,
            )
      
      return output

def prepare_data(_dataset_name: str):

    
    _dataset = pt.get_dataset(info_dict[_dataset_name]['path'])
    _queries = _dataset.get_topics('text')
      
    return _queries

def get_examples(_qid, _qv_df, _k):

    qv_examples = ''
    for number, rqText in enumerate(_qv_df[_qv_df.qid==_qid].sort_values(by=['rank'], ascending=True).iloc[:_k].rqText):
        qv_examples += f'Example {number}: {rqText}\n'
    # print('[debug]', qv_examples)
    return qv_examples

def gen_kshot_qv(qid:str, qText:str, _qv_df, _k):

    output = llama_call(llm, construct_kshot_prompt(qText, get_examples(qid, _qv_df, _k)), temperature=0.3)
    output_text = output['choices'][0]['text']
    try:
        generated_qvs, success = json5.loads(output_text), True
    except Exception as e1:
        try:
            match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if match:
                output_text = match.group(0)
                generated_qvs, success = json5.loads(output_text), True
            else:
                generated_qvs, success = output_text, False
                print("No JSON object found in output")
        except Exception as e2:
            generated_qvs, success = output_text, False
            print("Raw parse error:", e1)
            print("Regex-extracted parse error:", e2)
        # generated_qvs, success = output['choices'][0]['text'], False
        # print(e)
        # print('[debug]', output['choices'][0]['text'])
    return generated_qvs, success
  
def construct_0shot_prompt(qText):
    preamble = "You are an experienced searcher. Please reformulate the following query in 10 different ways so the reformulated queries hava similar (either more specific or more generic) information needs as the original one. "
    preamble += "Please put the reformulated queries in a json structure, such as {'Q_i': <reformulated query>}, where i is an integer between 0 and 9. "
    preamble += "End your answer after the reformulation immediately with </response>\n"
    postamble = "<response>"
    
    return f'{preamble}\n<query>{qText}<\query>\n{postamble}\n'

def construct_kshot_prompt(qText, examples):
    preamble = "You are an experienced searcher. Please reformulate the following query in 10 different ways so the reformulated queries hava similar (either more specific or more generic) information needs as the original one. "
    preamble += "Please reference the provided examples of real-life queries while reformulating the query. "
    preamble += "Please put the reformulated queries in a json structure, such as {'Q_i': <reformulated query>}, where i is an integer between 0 and 9. "
    preamble += "End your answer after the reformulation immediately with </response>\n"
    postamble = "<response>"

    print('[debug]', f'{preamble}\n<query>{qText}<\query>\n<example_queries>\n{examples}<\example_queries>\n{postamble}\n')
    
    return f'{preamble}\n<query>{qText}<\query>\n<example_queries>\n{examples}<\example_queries>\n{postamble}\n'

def gen_0shot_qv(qText: str):

    output = llama_call(llm, construct_0shot_prompt(qText), temperature=0.3)
    try:
        generated_qvs, success = json5.loads(output['choices'][0]['text']), True
    except:
        generated_qvs, success = output['choices'][0]['text'], False
        print('[debug]', output['choices'][0]['text'])
    return generated_qvs, success

def update_json_result_file(file_name, result_to_write):
    f = open(file_name, "w+", encoding='UTF-8')
    json.dump(result_to_write, f, indent=4)
    f.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='dl_19', choices=['dl_19', 'dl_20', 'dl_21', 'dl_22', 'webis-touche2020', 'trec_covid'])
    parser.add_argument("--q_retriever", type=str, default='bm25', choices=['bm25', 'sbert', 'dragon', 'tct', 'dragon_qasd', 'tct_qasd'])
    parser.add_argument("--hop_num", type=int, default=1, choices=[1, 2])
    parser.add_argument("--p", type=int, default=0)
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    q_retriever = args.q_retriever
    hop_num = args.hop_num
    p = args.p # number of retrieved examples

    info_dict = {'dl_19': {'path': 'irds:msmarco-passage/trec-dl-2019/judged'}, 'dl_20': {'path': 'irds:msmarco-passage/trec-dl-2020/judged'}, 
                 'dl_21': {'path': 'irds:msmarco-passage-v2/trec-dl-2021/judged', 'meta': {'docno': 64, 'title': 100, 'text': 4096}}, 'dl_22': {'path': 'irds:msmarco-passage-v2/trec-dl-2022/judged', 'meta': {'docno': 64, 'title': 100, 'text': 4096}},
             'webis-touche2020': {'path': 'irds:beir/webis-touche2020/v2', 'meta': {'docno': 64, 'title': 100, 'text': 4096, 'url': 256, 'stance':16}},
             'trec_covid': {'path': 'irds:beir/trec-covid', 'meta': {'docno': 64, 'title': 100, 'text': 4096}},
            }

    print('loading llm')
    llm = load_llama()
    print('loading queries')
    queries = prepare_data(dataset_name)

    if(p == 0):
        qv_df = 0
    else:
        qv_df = pd.read_csv(f'./qv_res/reranked_{dataset_name}_{q_retriever}_{hop_num}hop.csv')
        qv_df.qid = qv_df.qid.astype('str')

    print(f'[now at] {dataset_name} {q_retriever} {hop_num}hop {p}')
    
    if(p == 0):
        output_dir = f'./gen_qv_res/{dataset_name}_0shot_qvs'
        path = Path(f'{output_dir}.json')
        if path.exists():
            print("File exists")
            raise RuntimeError("don't need to continue")
        else:
            pass

        queries[['gen_qvs', 'success_generated']] = queries['query'].apply(pd.Series(gen_0shot_qv))
        queries.to_csv(f'{output_dir}.csv', index=False)

        qv_total_dict = {}

        for row in queries.iterrows():
            qid, qText, gen_qvs = row[1]['qid'], row[1]['query'], row[1]['gen_qvs']
            qv_total_dict.update({qid: {'query': qText, 'gen_qvs': gen_qvs}})

        with open(path, 'w') as f:
            json.dump(qv_total_dict, f)
            
    else:
        output_dir = f'./gen_qv_res/{dataset_name}_{p}shot_{hop_num}hop_{q_retriever}_qvs'
        path = Path(f'{output_dir}.json')
        if path.exists():
            print("File exists")
            raise RuntimeError("don't need to continue")
        else:
            pass
        
        queries[['gen_qvs', 'success_generated']] = queries.apply(lambda x: pd.Series(gen_kshot_qv(x['qid'], x['query'], qv_df, p)), axis=1)
        queries.to_csv(f'{output_dir}.csv', index=False)
            
        qv_total_dict = {}
            
        for row in queries.iterrows():
            qid, qText, gen_qvs = row[1]['qid'], row[1]['query'], row[1]['gen_qvs']
            qv_total_dict.update({qid: {'query': qText, 'gen_qvs': gen_qvs}})
            
        with open(path, 'w') as f:
            json.dump(qv_total_dict, f)
