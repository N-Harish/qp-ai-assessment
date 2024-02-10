# Chatbot to answer OS related Engineering Questions Using LangChain


* The chatbot takes user inputs and then retrieves 3 relavant documents by using similarity search (The threshold was set to 0.75)
* FAISS vector db was used to store the embeddings
* If the requested question was not present in the document, Bot says I don't know the answer
* For text embedding, Instruct Embedding was used (Since to use OpenAIs Embedding for long context we need to pay, It ws not used here)
* Google Flan-t5-large llm was used to generate response


## How to run

* Create a conda environment with python3.9
    ```
    conda create -n env_name pyython=3.9
    ```
    (Note:- Replace env-name with your environment name)

* Run ```conda activate env_name``` to activate the environment  (Note:- Replace env-name with your environment name)

* Run ```pip install -r requirements.txt``` to install the requirements

* Generate an API Token from huggingface. ( [Refer this url](https://huggingface.co/docs/hub/en/security-tokens) )
* Replace HF_TOKEN in constants.py with this API Token
* Run the App using 
    ```
    streamlit run chatbot.py
    ```

* vector.pkl file will be created automatically on running the code initially. If you don't want to create it the you can donload the vector file from [this link](https://drive.google.com/drive/folders/19mQjVJpX2llzbJ2l-Ezg5Wy5w6Sua6iG?usp=drive_link)

## Output

![Output](./RaG-chatbot1.png?raw=true "Output")

![Output](./RaG-chatbot2.png?raw=true "Output")