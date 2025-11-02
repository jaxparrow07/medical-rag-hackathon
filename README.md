# Medical RAG ( for Hackathon )
This is something we built for a hackathon ( VIT Chenani - hack-a-cure ) in just around 12 hours. Obviously, it's AI assisted 'cause we didn't want to spend time fixing syntax errors or programming logics since its limited by time. The evaluation was with RAGAS and we ended up getting around 0.62~ ( out of 1 where faithfulness, correctness, context retrieval would be considered ).<br><br>

The one we built for the hackathon with no furuther changes is available at the `legacy` branch. This branch is a refactored and fixed version of it with further improvements. The legacy branch was not bad per se, but it wasn't tailored for medicine related queries nor did it have any query reformulation due to the limited time.

### Getting started
Make sure you have python installed ( preferably something older than the latest if you don't want to compile them yourself ).

#### Step 0: Clone this repo
```bash
git clone https://github.com/jaxparrow07/medical-rag-hackathon.git
cd medical-rag-hackathon
```

#### Step 1: Create a Virtual Env and Install requirements - Recommended but optional
```bash
python3 -m venv rag
```
```bash
./rag/bin/activate{.fish,.ps...}
```
This ensures that all the packages are handled by this specific projects virtual environment and that it doesn't conflict with your global packages. Then you can proceed to install the dependencies.
```bash
pip3 install -r requirements.txt
```

#### Step 2: Configuring .env files and models
Copy the contents of `.env.example` into a `.env` file and place your API Keys. If you wish to change to a different model for **query reformulation**, **embedding (not recommended)**, or even **generation**. You can do that in the `src/config.py`. It houses all the important configurations.<br><br> If you encounter any spacy related issues. Try to download this model manually.
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

#### Step 3: Creating a vector database of your PDFs
Place your PDF files in the `data/raw_pdfs`. By default, this only performs basic text extraction and cleanup. If you want to extract tables you have to configure the `src/config.py` to allow `extract_tables` in the `ENHANCED_FEATURES_CONFIG`.**NOTE: Extracting tables is a slow proces. Use only if important data is represented in tables.**<b>r<br>

Running the following command
```bash
python3 setup_database.py
```
It'll take a while to store them in the vector db. If you see any errors regarding memory. Configure the `config.py` at `EMBEDDING_CONFIG` for the key `batch_size` and try to halve it.

#### Step 4: Running the RAG
```bash
python3 main.py
```
This command downloads all the required models and starts the interactive tui. You can input your query and see it verbosely process the queries.


### Issues in the `legacy` branch
- **Incorrect embedding Model**: The embedding model that was used was not a medicine specific model. So, it couldn't link between natural language and professional term of the same topic which is crucial.

- **Not using query reformulation**: Not all the questions were direct. So, reformulating the query into a statement of different variations would've yieleded proper results.

- **Word based chunking instead of semantic chunking**: Important information was cutoff due to word based chunking resulting in conclusions and other crucial details not being returned even though their parent sentence was.
