import os
import json
import requests

from fastapi import FastAPI, Request, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware

from haystack.document_stores import ElasticsearchDocumentStore
from haystack import Pipeline
from haystack.nodes import TextConverter, PreProcessor
from haystack.nodes import BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader


# Documents must be .txt files
doc_dir = '/documents/data/'


def index_documents(doc_store: ElasticsearchDocumentStore):
    # Create pipeline
    indexing_pipeline = Pipeline()
    text_converter = TextConverter()
    preprocessor = PreProcessor(
        clean_whitespace=True,
        clean_header_footer=True,
        clean_empty_lines=True,
        split_by="word",
        split_length=400,
        split_overlap=20,
        split_respect_sentence_boundary=True,
    )
    indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
    indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter"])
    indexing_pipeline.add_node(component=doc_store, name="DocumentStore", inputs=["PreProcessor"])
    # Gather recursively files to index
    files_to_index = [
        os.path.join(dir_path, f)
        for dir_path, _, filenames in os.walk(doc_dir)
        for f in filenames
        if os.path.splitext(f)[1] == '.txt'
    ]
    # Run indexing
    indexing_pipeline.run_batch(file_paths=files_to_index)


# reader = FARMReader(model_name_or_path="/models/medBIT-r3-plus_75/", use_gpu=False)
reader = FARMReader(model_name_or_path="IVN-RIN/medBIT-r3-plus", use_gpu=False)

document_store = ElasticsearchDocumentStore(
    host='hbd_elasticsearch', # 'host.docker.internal',
    port=9200,
    username="",
    password=""
)

# Check whether to index documents
if document_store.count_documents() == 0 and os.path.exists(doc_dir):
    index_documents(document_store)

retriever = BM25Retriever(document_store=document_store)

pipe = ExtractiveQAPipeline(reader, retriever) 

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
     "http://localhost:8080",
     "http:\/\/131\.175\.15\.22:61111\/hbd-demo\/*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




from typing import Dict, List
# This will act as our "cache"
#cache: Dict[str, List[Dict]] = {}

outputs_list = []


@app.post('/patient_search')
async def patient_search(request: Request):
    request_data = await request.json()
    query = request_data['query']
    prediction = pipe.run(
        query=query,
        params={
            "Retriever": {"top_k": 10},
            "Reader": {"top_k": 10}
        }
    )

    outputs = []
    for document in prediction['documents']:
        output_dict = {}
        output_dict['text'] = document.content
        output_dict['document_score'] = document.score
        output_dict['document_id'] = document.id[0:5]
        for answer in prediction['answers']:
            # print(answer.document_ids[0])
            if (answer.document_ids[0] == document.id):
                output_dict['context'] = answer.context
                output_dict['answer'] = answer.answer
        outputs.append(output_dict)
        
    # Store the results in the cache
    outputs_list.append(outputs)
    #cache[query] = outputs
    return { 'output': outputs}

############################################################################
def ask_to_llm(prompt, system_message="You are a helpful assistant and a medical expert"):
        data = {
        "messages": [
            {
                "content": system_message,
                "role": "system"
            },
            {
                "content": prompt,
                "role": "user"
            }
        ],
        'temperature': 0,
        'max_tokens': 500,
        # 'mirostat_tau': 0.0
        }
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        response = requests.post(
            " http://hbd_llamacpp:8000/v1/chat/completions/",
            #"http://host.docker.internal:51124/v1/chat/completions",
            headers=headers,
            json=data
        )
        if response.ok:
            completion_json = json.loads(response.text)
            completion_text = completion_json['choices'][0]['message']['content']
            return completion_text
        else:
            raise ValueError(f"Request problem. Status Code: {response.status_code}")
        
@app.post('/criteria_check')
async def criteria_check(request: Request):
    request_data = await request.json()
    criteria = request_data['criteria']
    #query = request_data['query']
    #patient_search_output = cache.get(query, [])
    # patient_search_output = await patient_search(request)
    patient_search_output = outputs_list[-1] if outputs_list else []
    i_criteria = []
    for document in patient_search_output:
        prompt = f'''Given the following clinical text:
        ```
        {document['text']}
        ```
        and given the following eligibility inclusion criteria for a clinical trial:
        ```
        {criteria}
        ```
        Determine whether the patient would be elegible for the clinical trial, responding either "Elegible, because ...", "Potentially elegible, if ..." or "Not elegible, because ..."  
        Response: 
        '''
        # If it contains multiple inclusion criterias, re-write those criterias in the following format:
        # Criterion 1 - …
        # Criterion 2 - …
        # …
        # Criterion N - …
        # Then answer with the format like: YES, it is
        # '''
        answer = ask_to_llm(prompt)
        i_criteria.append(answer)
    # i_criteria = ["yes", "no", "yes"]
    in_criteria = []
    for document in i_criteria:
        output_dict = {}
        output_dict['text'] = document
        in_criteria.append(output_dict)
    
    return {'criteria': in_criteria}
