
---

# Health Query - Powered by Llama 3B

## Description
The Intelligent Health Query Assistant is a project that leverages the power of the Llama 3B NLP generative model, fine-tuned specifically on a medical dataset related to HIV. This model is fine tuned to answer questions related to HIV disease, By providing query model can generate response of query.
## Project workflow 
![llama_flow drawio](https://github.com/ioptime-official/healthquery_llama/assets/72988974/0c0d12b7-37da-40b7-bf53-7ce81ca890d4)

<p align="justify">In this project, we begin by extracting relevant data pertaining to HIV from a diverse array of medical resources, including eHealth Forum, iCliniq, Question Doctors, and WebMD. With our data collected, we proceed to select the LLama 3B model for fine-tuning, a sophisticated NLP generative model. We then ensure that our extracted data is properly formatted into a template that aligns with the model's input requirements. To make the fine-tuning process accessible even on low-end hardware, we employ the LoRA (Low-Resource Adaptation) technique.
The core of our project involves training the model using this specialized dataset. Once the training is complete, the fine-tuned model files are available for download. For testing purposes, we load the base 3B model, and to enhance its capabilities, we integrate the LoRA fine-tune files. With the model primed and ready, we can now harness its power to make accurate predictions and respond to inquiries related to HIV disease. This comprehensive approach ensures the model's effectiveness in providing valuable insights into HIV-related queries.</p>

## Model Information
- **Base Model**: RedPajama-INCITE-Base-3B-v1
- **Developed by**: Together and leaders from the open-source AI community, including Ontocord.ai, ETH DS3Lab, AAI CERC, Université de Montréal, MILA - Québec AI Institute, Stanford Center for Research on Foundation Models (CRFM), Stanford Hazy Research research group, and LAION.
- **Fine-Tuning**: The LLama 3B model was fine-tuned on a custom medical text dataset specific to HIV for improved performance in generating responses to HIV-related questions.
- **LoRA Implementation**: Using lora during training because, it enhances model efficiency with LoRA (Loose Random Attention) technology, and prepares it for training with reduced precision, typically using 8-bit integers instead of higher-precision floating-point numbers. The LoRA configuration includes specific parameters like the number of attention heads (r=16) and an alpha value (lora_alpha=32) to fine-tune the model's behavior. This combination of techniques optimizes the model's performance while conserving computational resources, making it well-suited for various demanding natural language processing tasks.

## Dataset Information
- **Dataset Sources**: Data related to HIV was extracted from various medical sources, including:
  - eHealth Forum
  - iCliniq
  - Question Doctors
  - WebMD
- **Dataset Size**: After extraction, the dataset contains a total of 3,253 rows.
- **Data Preprocessing**: Data preprocessing was performed to format the dataset in a way that the model accepts it. The model expects data in "instruction" and "output" cloumn, in JSON format, so the dataset was transformed accordingly.
- **Sample input data** : Before fitting data to model the data should be in following format.
 ```bash
  def generate_prompt(data_point):
  return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  {data_point["instruction"]}

  ### Response:
  {data_point["output"]}"""
  ```
```python
print(generate_prompt(Hiv_Data['my_dataset'][5]))
```
```
 ### Instruction:
Sex Male Age 28 Height 6 ft Weight 196 Lbs no medical conditions and no medication Hi Docs, so a month ago I was vacationing in Cuba, on my last day I got stung by a sea urchin in my right foot. During my flight back in the plane, I started getting strong intense muscle pains in my right arm starting from the wrist to shoulder, a pain so intense it felt like my arm was going to explode. This pain went on for 2 days waking me up screaming and crying at night, so I went to the ER, the doctor says I have a shoulder inflammation and gave me naproxen. 

### Response:
The sting introduced some bacteria that caused an initial infection, leading to the post-infectious inflammatory response even after the bacteria were addressed. The blurry vision and leg numbness are concerning & make me think there is something neurological or maybe vascular happening. Toxins from the sting might be affecting nerve function. You have seen an infectious disease specialist, I would next go to a neurologist.
```
## Installation and Dependencies
To run the fine-tuned model, we need to install the necessary libraries and dependencies.following are the commands:

```bash
# Install required libraries
pip install transformers
pip install torch
pip install -Uqq  git+https://github.com/huggingface/peft.git
pip install -Uqq transformers datasets accelerate bitsandbytes
```

## Usage Instructions
To use the fine-tuned model for answering HIV-related questions, follow these instructions:
- fine tuned model can be download [here](https://github.com/ioptime-official/healthquery_llama/tree/main/model_Files)

```python
# Load the base model
from transformers import AutoTokenizer, AutoModelForCausalLM
install_dependencies()
model = '7' # Pick base model'


if model == '7B':
    model_name = ("togethercomputer/RedPajama-INCITE-Base-7B-v0.1","togethercomputer/RedPajama-INCITE-Base-7B-v0.1")
    run_name = 'redpj7B-lora-int8-alpaca'
    dataset = 'johnrobinsn/alpaca-cleaned'
    peft_name = 'redpj7B-lora-int8-alpaca'
    output_dir = 'redpj7B-lora-int8-alpaca-results'
else: #3B
    model_name = ("togethercomputer/RedPajama-INCITE-Base-3B-v1","togethercomputer/RedPajama-INCITE-Base-3B-v1")
    run_name = 'redpj3B-lora-int8-alpaca'
    dataset = 'johnrobinsn/alpaca-cleaned'
    peft_name = 'redpj3B-lora-int8-alpaca'
    output_dir = 'redpj3B-lora-int8-alpaca-results'

model_name[1],dataset,peft_name,run_name
# libaraies for loading trained model
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
# load base LLM model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name[0],
    load_in_8bit=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name[1])
tokenizer.pad_token_id = 0
tokenizer.add_special_tokens({'eos_token':''})

model.eval()
# model prompt template.
def generate_prompt(data_point):
  return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:"""

# Response generation function.
def generate(instruction,input=None,maxTokens=256):
    prompt = generate_prompt({'instruction':instruction,'input':input})
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=maxTokens,
                             do_sample=True, top_p=0.9,pad_token_id=tokenizer.eos_token_id,
                             forced_eos_token_id=tokenizer.eos_token_id)
    outputs = outputs[0].tolist()
    # Stop decoding when hitting the EOS token
    if tokenizer.eos_token_id in outputs:
        eos_index = outputs.index(tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[:eos_index])
        # Don't show the prompt template
        sentinel = "### Response:"
        sentinelLoc = decoded.find(sentinel)
        if sentinelLoc >= 0:
            print(decoded[sentinelLoc+len(sentinel):])
        else:
            print('Warning: Expected prompt template to be emitted.  Ignoring output.')
    else:
        print('Warning: no  detected ignoring output')

# Load the trained model
peft_model_id = '/content/drive/MyDrive/llama(hiv)'# Uncomment to use locally saved adapter weights if you trained above
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
model.eval()
print("Peft model adapter loaded")

# Generate responses to HIV-related questions
question = "what are the main symptoms of HIV?"
torch.manual_seed(42)
x=generate(x,maxTokens=300)
print(response)
```

## Model Output
- Question:what are the main symptoms of HIV?
> the main symptoms of HIV include tiredness, flu-like illness (feeling generally unwell with high temperature or shivering), sore throat, muscle and joint pain, flu-like illness, headache, mouth ulcers or sores, mouth ulcers and skin changes that may be fluid-filled, especially around the nose or in the genital area (vulva or penis).

## References
- https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1




---
