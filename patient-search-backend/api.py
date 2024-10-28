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
if document_store.get_document_count() == 0 and os.path.exists(doc_dir):
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


def ask_to_llm(prompt, system_message="You are a healthcare assistant and a medical expert on information extraction"):
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

    patient_search_output = outputs_list[-1] if outputs_list else []
    i_criteria = []

    few_shot_examples = '''
   
    Eligibility Criteria:
    - Age 40 to 60
    - Female
    - History of  urinary tract infections
 
    Clinical Text:
    ```
    Medical History:
    A 30-year-old woman presented with recurrent urinary tract infections, occasional left lumbar pain, and continuous dribbling of urine. Physical examination revealed two urethral openings, one below the other in the sagittal plane, below the clitoris. Her renal function tests were normal.
 
    Diagnostic Imaging:
    Ultrasound scan of the abdomen showed gross hydroureteronephrosis of the left kidney extending up to the lower ureter, and the urinary bladder had a very large capacity with irregular lobulated contours. Intravenous pyelogram (IVP) showed a large capacity bladder with left gross hydroureteronephrosis and severely tortuous left ureter suggestive of left obstructive megaureter. Micturating cystourethrogram also revealed a very large capacity bladder. The magnetic resonance imaging of the spine was normal.
 
    Cystoscopy:
    Cystoscopy showed a double urethra, one above the other in the sagittal plane. The posterior urethra was in the orthotopic position and was opening into the large capacity bladder. The anterior urethra was present just posterior to the clitoris. The internal opening of the accessory (anterior) urethra was located about 2 cm cranial to the bladder neck on the anterior wall of the urinary bladder. Green arrow in shows the vaginal opening.
 
    Treatment:
    Excision of the accessory anterior urethra with reduction cystoplasty and left ureteric tailoring with reimplantation was performed in a single setting. The excision of the accessory urethra was performed with the child in the lithotomy position through the abdominoperineal route. Since the excised urethra was posterior to the clitoris, the nerve supply to the clitoris was not disrupted. The orthotopic urethra and the bladder neck were also safeguarded during the dissection. The resected accessory urethra had transitional cell lining with thin, poorly muscular wall on the histopathology.
 
    Follow-Up:
    The patient was followed up with 3 monthly urine microscopy and culture, RFT, ultrasound scan of the abdomen at 6 months, and IVP at the end of 1 year of surgery. Since these were normal, she was followed up with only RFT and ultrasound scan of the abdomen annually for the past 5 years and is doing well.
    ```
    Response:
    Eligible because the patient is a 30-year-old woman with recurrent urinary tract infections which meets all criteria.
   
    Clinical Text:
    ```
    Discharge Summary:
 
    Patient: 81-year-old man with metastatic melanoma, remote history of mantle cell lymphoma, mild oral lichen planus, gout, gastroesophageal reflux disease, chronic bronchitis, benign prostatic hyperplasia, and depression.
 
    History: The patient failed first-line course of BRAF-targeted therapy with dabrafenib and trametinib. In November 2018, he started with second-line pembrolizumab, which was discontinued in May 2019 due to the serial progression of disease on imaging. Within 6 weeks of discontinuation of therapy, the patient presented with a generalized skin eruption, composed of multiple scaly erythematous papules and plaques, some with central hyperkeratotic crust, in a symmetric distribution on the arms, chest, back, abdomen, and upper legs. No other medications were introduced prior to development of the cutaneous eruption.
 
    Treatment: The patient was initially treated with high-dose potency topical steroids and rapidly moved to prednisone 1 mg/kg when he did not have an adequate response. He had a mild initial good response, but the lesions flared on the taper of steroids at the same time as a grade 2 immune-related colitis. On increase of corticosteroids, the patient had a resolution of his colitis and stability of his lesions; prednisone was tapered for a total of 2 months of treatment. He was subsequently serially treated with methotrexate for 4 weeks, followed by mycophenolate mofetil for 4 weeks, and then cyclosporine for 4 weeks, with a mild response to therapy at best. After consultation in a multidisciplinary immune-toxicity team, it was decided that NBUVB phototherapy could be considered. The patient received 17 total sessions, which induced a significant healing of his lesions and the resolution of the pruritus and skin discomfort.
 
    Outcome: Subsequent to his immune-toxicities, a significant systemic antitumor response was noted on his staging scans, which are currently ongoing over 12 months after cessation of PD-1 therapy.
 
    Impression: The patient's cutaneous eruption was diagnosed as a lichenoid drug reaction, which was effectively treated with NBUVB phototherapy. He had a mild response to other systemic treatments. The patient was discharged in stable condition with recommendations for follow-up with his oncologist.
    ```
    Response:
    Not Eligible because age is 81 years old which is outside the range of 40 to 60.
   
    Clinical Text:
    ```
    Electronic Health Record
 
    Patient Name: [De-identified]
    Age: [AGE]
    Medical Record Number: [De-identified]
 
    Discharge Summary
 
    Admission Date: [Date]
    Discharge Date: [Date]
 
    Admitting Diagnosis: Abdominal pain and a palpable mass in the right hemiabdomen
 
    Hospital Course:
    The patient presented with abdominal pain and a palpable mass in the right hemiabdomen. The physical examination was otherwise normal. Abdominal CT revealed an oval, sharply marginated, heterogeneous, mass-like lesion in the wall of the right colon that was 110 × 95 × 90 mm in size. A postcontrast CT examination of the tumour showed heterogeneous enhancement. On MRI, the tumour displayed heterogeneous signal intensity on T2-weighted images with irregular areas of hyperintensity.
    Surgery was performed and a well-marginated mass that invaded the wall of the caecum and small bowel mesentery near the terminal ileum was found intraoperatively. Thus, a right hemicolectomy with partial resection of the terminal ileum was performed, with latero-lateral ileocolonic anastomosis. Histological examination revealed mesenchymal proliferation with uniform bland spindle-shaped cells in long ‘sweeping’ fascicles or loose, vague storiform arrays. Medium-sized stromal blood vessels with some perivascular hyalinization and smaller vessels often elongated and compressed between lesional fascicles were found. Further immunohistochemical examination showed nuclear immunohistochemical identification of beta-catenin and the diagnosis of DTF was made.
 
    Discharge Diagnosis: DTF (Desmoid-type fibromatosis)
 
    Discharge Condition: Stable
 
    Summary:
    The patient presented with abdominal pain and a palpable mass in the right hemiabdomen. The diagnosis of DTF was made after surgery and histological examination. The patient underwent a right hemicolectomy with partial resection of the terminal ileum, and latero-lateral ileocolonic anastomosis was performed. At 1 year after surgery, the patient remains disease-free and has no signs of disease recurrence. The patient was discharged in stable condition.
 
    Recommendations:
    The patient should continue regular follow-up with their healthcare provider in order to monitor their condition and detect any potential disease recurrence.
 
    Follow-up:
    The patient will have regular follow-up appointments with their healthcare provider.
    ```
    Response
    Insufficient information because age is not mentioned in the Clinical Text.
    '''

    for document in patient_search_output:
        prompt = f'''{few_shot_examples}
       
        Evaluate the patient's eligibility based on the clinical text and the provided inclusion criteria.
       
        clinical text:
        ```
        {document['text']}
        ```
        Inclusion Criteria:
        ```
        {criteria}
        ```
       
        Use the following guidelines to determine eligibility:
        - If the required information is available in clinical text, use the exact values in your response.
        - If any single criterion is not met, respond with "Not Eligible because ..." and state the reason.
        - If the information is missing, ambiguous, inferred, or unclear in the clinical text, respond with "Insufficient information because ..." and explain that the necessary information is not explicitly stated.
        - Note that placeholders like "[REDACTED]", "[AGE]", "[DOB]" or any combinations like "Patient Age: [REDACTED]" are not considered valid age information and should not be used.
        - If a criterion is inferred (e.g., the patient's age is inferred but not explicitly mentioned), respond with "Insufficient information because ..." and explain that the information is not explicitly stated.
        - If any numeric values was not mentioned or mentioned as "[REDACTED]" (e.g., date of birth: [REDACTED]), respond with "Insufficient information because ...".
        - Only if all criteria are met exactly, respond with "Eligible because ...".
        '''

        answer = ask_to_llm(prompt)
        i_criteria.append(answer)

    comparison_results = []
    for response in i_criteria:
        reformat_prompt = f'''
        Organize the following eligibility response into a concise, structured format.
 
        Eligibility Response
        ```
        {response}
        ```
 
        Respond in this format:
        [Eligibility Status] because: [Concise reason]
       
        Eligibility Status: Eligible, Not Eligible, or Insufficient information
        Respond only in the format above without extra text.
        '''

        # Second step: Ask LLM to perform the comparison and return eligibility result
        comparison_result = ask_to_llm(reformat_prompt)
        comparison_results.append(comparison_result)

    return {'criteria': comparison_results}
