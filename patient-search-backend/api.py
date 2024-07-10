from fastapi import FastAPI, Request, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
import fitz
from fastapi.responses import FileResponse
from io import BytesIO
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader
import json
import requests

# reader = FARMReader(model_name_or_path="/models/medBIT-r3-plus_75/", use_gpu=False)
reader = FARMReader(model_name_or_path="IVN-RIN/medBIT-r3-plus", use_gpu=False)

document_store = ElasticsearchDocumentStore(
    host='elasticsearch', # 'host.docker.internal',  
    port=9200,
    username="",
    password=""
)

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
            "http://131.175.15.22:61111/patient-search-server/llama-server/v1/chat/completions/",
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
############################################################################

@app.post('/compute_output')
async def answer_question_list(request: Request):
    request_data = await request.json()
    input_text = request_data['input_text']
    output_text = generate_output(input_text)
    return {'output_text': output_text}


def IntersecOfSets(array):
    s1 = set(array.pop(0))
    s2 = set(array.pop(0))
    result = s1.intersection(s2)
    if len(array) > 2:
      for element in array:
        result = result.intersection(element)

    # Converts resulting set to list
    final_list = list(result)
    return final_list

@app.post('/convert_pdf')
async def convert_pdf(uploaded_pdf: UploadFile):
    print("FILE: ", uploaded_pdf.filename)
    with fitz.open(stream=BytesIO(uploaded_pdf.file.read())) as document:
        if len(document) > 2:
            # ----- FIND DUPLICATES ----- #
            all_elements = []
            for page in document:
                elements = []
                for area in page.get_text('blocks'):
                    box = fitz.Rect(area[:4])
                    if not box.is_empty:
                        elements.append(area[4])
                all_elements.append(elements)
        duplicates = IntersecOfSets(all_elements) if len(document) > 2 else [] # all the elements that are in common within all pages

        # ----- REMOVE DUPLICATES AND CLEAN TEXT ----- #
        dirty_text = []                                             # collect the "dirty" to compare the line with the previous one
        clean_text = ''                                             # put the clean text
        header_text = ''                                            # put the header text
        header_missing = True                                       # when False because we are at page 2 and collected everything
        for page in document:
            all_linetext = []                                       # use to remove duplicate lines
            for element in page.get_text('blocks'):
                header = False                                      # True when text is part of the header 
                box = fitz.Rect(element[:4])
                if not box.is_empty:
                    if element[4] in duplicates:                     # check if normal text or header text
                        header = True
                    linetext = page.get_textpage(box).extractWORDS() # get all the single words (inside the box) and their positioning 
                    if linetext not in all_linetext:
                        all_linetext.append(linetext)
                        if linetext != []:
                            dirty_text.append(linetext[0])
                            if header_missing and header: 
                                header_text += linetext[0][4] + ' '
                            elif not header:
                                if linetext[0][4] == 'Etichetta' and linetext[0][4] + ' ' + linetext[1][4] == 'Etichetta paziente':
                                    del linetext[0:2]
                                if linetext == []:
                                    continue
                                if linetext[0][4].isupper():
                                  clean_text += '\n' + linetext[0][4] + ' '
                                else:
                                  clean_text += linetext[0][4] + ' '
                        for i, line in enumerate(linetext[1:]): 
                            if line[4] == 'Etichetta' and line[4] + ' ' + linetext[i+2][4] == 'Etichetta paziente':
                                continue
                            if line[4] == 'paziente' and linetext[i][4] + ' ' + line[4] == 'Etichetta paziente':
                                continue

                            x1 = abs(dirty_text[-1][0] - line[0])
                            x2 = abs(dirty_text[-1][1] - line[1])
                            x3 = abs(dirty_text[-1][2] - line[2])
                            x4 = abs(dirty_text[-1][3] - line[3])
                            if (x1+x2+x3+x4 > 4):                      # compare the position of each word to remove duplicates
                                #check if we are on the same line, in terms of coordinates or text recognition 
                                if(dirty_text[-1][1] == line[1] and dirty_text[-1][3] == line[3]) or (dirty_text[-1][6] == line[6]):
                                    if header_missing and header: 
                                        header_text += line[4] + ' '
                                    elif not header: 
                                        clean_text += line[4] + ' '
                                else:
                                    line_break = line[6] - dirty_text[-1][6]
                                    if header_missing and header:
                                        header_text += min(2, line_break)*'\n' + line[4] + ' '
                                    elif not header:
                                        clean_text += line_break*'\n' + line[4] + ' '
                                dirty_text.append(line)

                        if header_missing and header: 
                            header_text += '\n\n' 
                        elif not header:
                            if clean_text[-1] == '.':
                                clean_text += '\n\n'
                            else:
                                clean_text += '\n'
            header_missing = False   

        # ----- ADD DUPLICATES AT THE END AND RETURN TEXT ----- #
        # clean_text += '\n\n ---------- HEADERS --------- \n'
        # clean_text += header_text

        return {'pdf_text': clean_text}
    
@app.post('/return_pdf')
async def return_pdf(uploaded_pdf: UploadFile):
    filename = './pymupdf.pdf'
    document =  fitz.open(stream=BytesIO(uploaded_pdf.file.read()), filetype='pdf')
    for page in document:
        for area in page.get_text('blocks'):
            box = fitz.Rect(area[:4])
            if not box.is_empty:
                page.add_rect_annot(box)
    
    output_pdf = BytesIO()                          
    document.save(filename)                         
    output_pdf.seek(0)                             
    return FileResponse(filename, filename='pymupdf.pdf')