################################
# Generated QVs using an LLM ###
# Backends: Gemini (API), #####
# OpenAI (API), or a local ####
# Llama GGUF file. #############
# The results are NOT re-ranked#
# by RBO. ######################
################################

import pyterrier as pt
import json
import pandas as pd
import sys
import os
import time
from itertools import product
from pathlib import Path
from typing import List
import argparse
import re

from pydantic import BaseModel

class ReformulatedQueries(BaseModel):
    reformulations: List[str]

def load_llama(model_path=None):
    from llama_cpp import Llama

    if model_path is None:
        cwd = os.getcwd()
        parent = os.path.dirname(cwd)
        model_path = os.path.join(parent, 'gguf_storage', 'Meta-Llama-3-8B-Instruct.Q8_0.gguf')
    print(model_path)
    llm = Llama(
        model_path=model_path,
        logits_all=False,
        verbose=False,
        n_gpu_layers=-1, # Uncomment to use GPU acceleration
        n_ctx=2048, # Uncomment to increase the context window
    )
    llm.set_seed(1000)
    return llm

def load_gemini(api_key=None):
    from google import genai

    return genai.Client(api_key=api_key) if api_key else genai.Client()

def load_openai(api_key=None):
    from openai import OpenAI

    return OpenAI(api_key=api_key) if api_key else OpenAI()

def openai_call(client, model, prompt, temperature=0.3):
    return client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=ReformulatedQueries,
        temperature=temperature,
    )

def _rate_limit_retry_delay(exc):
    try:
        for d in exc.details.get('error', {}).get('details', []):
            if d.get('@type', '').endswith('RetryInfo'):
                delay_str = d.get('retryDelay', '')
                if delay_str.endswith('s'):
                    return float(delay_str[:-1])
    except Exception:
        pass
    return None

def gemini_call(client, model, prompt, temperature=0.3, max_retries=20):
    from google.genai import errors as genai_errors

    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ReformulatedQueries,
                    "temperature": temperature,
                },
            )
        except genai_errors.ClientError as e:
            if e.code == 429 and attempt < max_retries - 1:
                delay = _rate_limit_retry_delay(e) or min(60, 2 ** attempt)
                print(f'[rate limited] attempt {attempt + 1}/{max_retries}, retrying in {delay:.1f}s...')
                time.sleep(delay + 1)  # small buffer past the server's suggested delay
            else:
                raise

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

def prepare_data(_dataset_name: str, queries_path: str = None):

    if queries_path is not None:
        with open(queries_path) as f:
            first_line = f.readline()
        if 'qid' in first_line and 'query' in first_line:
            # a runfile-style CSV (has a header with qid/query columns, e.g. ./res/dl_19_bm25.csv)
            return pd.read_csv(queries_path, dtype={'qid': str})[['qid', 'query']].drop_duplicates()
        # a plain headerless TSV of qid<TAB>query text
        return pd.read_csv(queries_path, sep='\t', header=None, names=['qid', 'query'], dtype=str)

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
    import json5

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
    
    return f'{preamble}\n<query>{qText}</query>\n{postamble}\n'

def construct_kshot_prompt(qText, examples):
    preamble = "You are an experienced searcher. Please reformulate the following query in 10 different ways so the reformulated queries hava similar (either more specific or more generic) information needs as the original one. "
    preamble += "Please reference the provided examples of real-life queries while reformulating the query. "
    preamble += "Please put the reformulated queries in a json structure, such as {'Q_i': <reformulated query>}, where i is an integer between 0 and 9. "
    preamble += "End your answer after the reformulation immediately with </response>\n"
    postamble = "<response>"

    print('[debug]', f'{preamble}\n<query>{qText}</query>\n<example_queries>\n{examples}</example_queries>\n{postamble}\n')
    
    return f'{preamble}\n<query>{qText}</query>\n<example_queries>\n{examples}</example_queries>\n{postamble}\n'

def gen_0shot_qv(qText: str):
    import json5

    output = llama_call(llm, construct_0shot_prompt(qText), temperature=0.3)
    try:
        generated_qvs, success = json5.loads(output['choices'][0]['text']), True
    except:
        generated_qvs, success = output['choices'][0]['text'], False
        print('[debug]', output['choices'][0]['text'])
    return generated_qvs, success

def construct_0shot_prompt_structured(qText):
    return (
        "You are an experienced searcher. Paraphrase the following query in 10 different ways, "
        "preserving its exact information need, scope, and intent (do not make it broader or "
        "narrower). Use different wording, phrasing, and/or synonyms than the original and than "
        "each other.\n\n"
        f"Query: {qText}"
    )

def construct_kshot_prompt_structured(qText, examples, template=None):
    if template is not None:
        return template.format(query=qText, examples=examples)
    return (
        "You are an experienced searcher. Paraphrase the following query in 10 different ways, "
        "preserving its exact information need, scope, and intent (do not make it broader or "
        "narrower). Use different wording, phrasing, and/or synonyms than the original and than "
        "each other. Reference the provided examples of real-life queries for tone and style "
        "while paraphrasing.\n\n"
        f"Query: {qText}\n\n"
        f"Example real-life queries:\n{examples}"
    )

def gen_kshot_qv_gemini(client, model, qid: str, qText: str, _qv_df, _k, prompt_template=None):
    prompt = construct_kshot_prompt_structured(qText, get_examples(qid, _qv_df, _k), prompt_template)
    response = gemini_call(client, model, prompt)
    if response.parsed is not None:
        generated_qvs = {f'Q_{i}': q for i, q in enumerate(response.parsed.reformulations)}
        return generated_qvs, True
    print("No parsed output from Gemini:", response.text)
    return response.text, False

def gen_0shot_qv_gemini(client, model, qText: str):
    prompt = construct_0shot_prompt_structured(qText)
    response = gemini_call(client, model, prompt)
    if response.parsed is not None:
        generated_qvs = {f'Q_{i}': q for i, q in enumerate(response.parsed.reformulations)}
        return generated_qvs, True
    print("No parsed output from Gemini:", response.text)
    return response.text, False

def gen_kshot_qv_openai(client, model, qid: str, qText: str, _qv_df, _k, prompt_template=None):
    prompt = construct_kshot_prompt_structured(qText, get_examples(qid, _qv_df, _k), prompt_template)
    response = openai_call(client, model, prompt)
    if response.output_parsed is not None:
        generated_qvs = {f'Q_{i}': q for i, q in enumerate(response.output_parsed.reformulations)}
        return generated_qvs, True
    print("No parsed output from OpenAI:", response.output_text)
    return response.output_text, False

def gen_0shot_qv_openai(client, model, qText: str):
    prompt = construct_0shot_prompt_structured(qText)
    response = openai_call(client, model, prompt)
    if response.output_parsed is not None:
        generated_qvs = {f'Q_{i}': q for i, q in enumerate(response.output_parsed.reformulations)}
        return generated_qvs, True
    print("No parsed output from OpenAI:", response.output_text)
    return response.output_text, False

def update_json_result_file(file_name, result_to_write):
    f = open(file_name, "w+", encoding='UTF-8')
    json.dump(result_to_write, f, indent=4)
    f.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='dl_19',
                         help="Dataset key used for output filenames. If not one of the built-in keys "
                              "(dl_19, dl_20, dl_21, dl_22, webis-touche2020, trec_covid), --queries_path "
                              "must be given.")
    parser.add_argument("--q_retriever", type=str, default='bm25', choices=['bm25', 'sbert', 'dragon', 'tct', 'dragon_qasd', 'tct_qasd'])
    parser.add_argument("--hop_num", type=int, default=1, choices=[1, 2])
    parser.add_argument("--p", type=int, default=0)
    parser.add_argument("--queries_path", type=str, default=None,
                         help="Optional path to a local queries source, overriding automatic ir_datasets topic "
                              "lookup for --dataset_name. Accepts either a runfile CSV with 'qid'/'query' columns "
                              "(e.g. ./res/dl_19_bm25.csv) or a plain TSV of 'qid<TAB>query text' with no header.")
    parser.add_argument("--backend", type=str, default='gemini', choices=['gemini', 'openai', 'llama'])
    parser.add_argument("--model_path", type=str, default=None,
                         help="[llama backend] Path to the GGUF model file. Defaults to "
                              "<parent-of-cwd>/gguf_storage/Meta-Llama-3-8B-Instruct.Q8_0.gguf if not given.")
    parser.add_argument("--gemini_api_key", type=str, default=None,
                         help="[gemini backend] API key. Defaults to the GEMINI_API_KEY/GOOGLE_API_KEY "
                              "env var if not given.")
    parser.add_argument("--gemini_model", type=str, default='gemini-2.5-flash')
    parser.add_argument("--openai_api_key", type=str, default=None,
                         help="[openai backend] API key. Defaults to the OPENAI_API_KEY env var if not given.")
    parser.add_argument("--openai_model", type=str, default='gpt-4.1-nano')
    parser.add_argument("--prompt_template_file", type=str, default=None,
                         help="[kshot, gemini/openai backends] Path to a text file containing a "
                              "custom prompt template with '{query}' and '{examples}' placeholders. "
                              "Defaults to the built-in prompt if not given.")
    args = parser.parse_args()

    prompt_template = None
    if args.prompt_template_file is not None:
        with open(args.prompt_template_file, 'r', encoding='UTF-8') as f:
            prompt_template = f.read()

    dataset_name = args.dataset_name
    q_retriever = args.q_retriever
    hop_num = args.hop_num
    p = args.p # number of retrieved examples

    info_dict = {'dl_19': {'path': 'irds:msmarco-passage/trec-dl-2019/judged'}, 'dl_20': {'path': 'irds:msmarco-passage/trec-dl-2020/judged'}, 
                 'dl_21': {'path': 'irds:msmarco-passage-v2/trec-dl-2021/judged', 'meta': {'docno': 64, 'title': 100, 'text': 4096}}, 'dl_22': {'path': 'irds:msmarco-passage-v2/trec-dl-2022/judged', 'meta': {'docno': 64, 'title': 100, 'text': 4096}},
             'webis-touche2020': {'path': 'irds:beir/webis-touche2020/v2', 'meta': {'docno': 64, 'title': 100, 'text': 4096, 'url': 256, 'stance':16}},
             'trec_covid': {'path': 'irds:beir/trec-covid', 'meta': {'docno': 64, 'title': 100, 'text': 4096}},
            }

    print(f'loading backend: {args.backend}')
    if args.backend == 'gemini':
        gemini_client = load_gemini(args.gemini_api_key)
        gemini_model = args.gemini_model
    elif args.backend == 'openai':
        openai_client = load_openai(args.openai_api_key)
        openai_model = args.openai_model
    else:
        llm = load_llama(args.model_path)
    print('loading queries')
    queries = prepare_data(dataset_name, args.queries_path)

    if(p == 0):
        qv_df = 0
    else:
        qv_df = pd.read_csv(f'./qv_res/reranked_{dataset_name}_{q_retriever}_{hop_num}hop.csv')
        qv_df.qid = qv_df.qid.astype('str')

    print(f'[now at] {dataset_name} {q_retriever} {hop_num}hop {p}')

    os.makedirs('./gen_qv_res', exist_ok=True)

    if(p == 0):
        output_dir = f'./gen_qv_res/{dataset_name}_0shot_paraphrase_qvs'
        path = Path(f'{output_dir}.json')
        if path.exists():
            print("File exists")
            raise RuntimeError("don't need to continue")
        else:
            pass

        if args.backend == 'gemini':
            queries[['gen_qvs', 'success_generated']] = queries['query'].apply(
                lambda x: pd.Series(gen_0shot_qv_gemini(gemini_client, gemini_model, x)))
        elif args.backend == 'openai':
            queries[['gen_qvs', 'success_generated']] = queries['query'].apply(
                lambda x: pd.Series(gen_0shot_qv_openai(openai_client, openai_model, x)))
        else:
            queries[['gen_qvs', 'success_generated']] = queries['query'].apply(lambda x: pd.Series(gen_0shot_qv(x)))
        queries.to_csv(f'{output_dir}.csv', index=False)

        qv_total_dict = {}

        for row in queries.iterrows():
            qid, qText, gen_qvs = row[1]['qid'], row[1]['query'], row[1]['gen_qvs']
            qv_total_dict.update({qid: {'query': qText, 'gen_qvs': gen_qvs}})

        with open(path, 'w') as f:
            json.dump(qv_total_dict, f)
            
    else:
        output_dir = f'./gen_qv_res/{dataset_name}_{p}shot_{hop_num}hop_{q_retriever}_paraphrase_qvs'
        path = Path(f'{output_dir}.json')
        if path.exists():
            print("File exists")
            raise RuntimeError("don't need to continue")
        else:
            pass
        
        if args.backend == 'gemini':
            queries[['gen_qvs', 'success_generated']] = queries.apply(
                lambda x: pd.Series(gen_kshot_qv_gemini(gemini_client, gemini_model, x['qid'], x['query'], qv_df, p, prompt_template)), axis=1)
        elif args.backend == 'openai':
            queries[['gen_qvs', 'success_generated']] = queries.apply(
                lambda x: pd.Series(gen_kshot_qv_openai(openai_client, openai_model, x['qid'], x['query'], qv_df, p, prompt_template)), axis=1)
        else:
            queries[['gen_qvs', 'success_generated']] = queries.apply(lambda x: pd.Series(gen_kshot_qv(x['qid'], x['query'], qv_df, p)), axis=1)
        queries.to_csv(f'{output_dir}.csv', index=False)
            
        qv_total_dict = {}
            
        for row in queries.iterrows():
            qid, qText, gen_qvs = row[1]['qid'], row[1]['query'], row[1]['gen_qvs']
            qv_total_dict.update({qid: {'query': qText, 'gen_qvs': gen_qvs}})
            
        with open(path, 'w') as f:
            json.dump(qv_total_dict, f)
