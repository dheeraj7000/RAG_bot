# %%
pip install langchain pinecone-client[grpc] tiktoken apache_beam openai

# %%
from datasets import load_dataset

data = load_dataset("Mutugi/housing", split='train[:1000]')
data

# %%
data.features

# %%
def addCols(data):
    data["title"] = "description:" + data["title"] + " | size:" + data["location"] + " | price:" + data["price"] + " | location:" + data["location"] 
    return data

# %%
ndata = data.map(addCols)

# %%
ndata = ndata.remove_columns(column_names=['price','size','location'])

# %%
import tiktoken

tiktoken.encoding_for_model('gpt-3.5-turbo')

# %%
import tiktoken

tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

tiktoken_len("hello I am a chunk of text and using the tiktoken_len function "
             "we can find the length of this chunk of text in tokens")

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

# %%
chunks = text_splitter.split_text(ndata[6]['title'])[:3]
chunks

# %%
tiktoken_len(chunks[0])

# %%
from getpass import getpass

OPENAI_API_KEY = "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"  # platform.openai.com

# %%
from langchain.embeddings.openai import OpenAIEmbeddings

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

# %%
texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here'
]

res = embed.embed_documents(texts)
len(res), len(res[0])

# %%
import pinecone

# find API key in console at app.pinecone.io
YOUR_API_KEY = "8c6fbaa9-444b-4acf-9a2c-a06ee8af52d5"
# find ENV (cloud region) next to API key in console
YOUR_ENV = "gcp-starter"

index_name = 'demo1'
pinecone.init(
    api_key=YOUR_API_KEY,
    environment=YOUR_ENV
)

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=len(res[0])  # 1536 dim of text-embedding-ada-002
    )

# %%
index = pinecone.GRPCIndex(index_name)

index.describe_index_stats()

# %%
from tqdm.auto import tqdm
from uuid import uuid4

batch_limit = 50

texts = []
metadatas = []

for i, record in enumerate(tqdm(ndata)):
    # first get metadata fields for this record
    metadata = {
        'id' : str(i),
        # 'location': record['location'],
        # 'size': record['size'],
        # 'price': record['price']
    }
    # now we create chunks from the record text
    record_texts = text_splitter.split_text(record['title'])
    # create individual metadata dicts for each chunk
    record_metadatas = [{
        "chunk": j, "text": text, **metadata
    } for j, text in enumerate(record_texts)]
    # append these to current batches
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)
    # if we have reached the batch_limit we can add texts
    if len(texts) >= batch_limit:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        texts = []
        metadatas = []

if len(texts) > 0:
    ids = [str(uuid4()) for _ in range(len(texts))]
    embeds = embed.embed_documents(texts)
    index.upsert(vectors=zip(ids, embeds, metadatas))

# %%
index.describe_index_stats()

# %%
from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# %%
query = "price of 2 bed house in runda"

vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
)

# %%
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# %%
qa.run(query)


