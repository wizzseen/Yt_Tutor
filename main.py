from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from datasets import Dataset
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from youtube_transcript_api import YouTubeTranscriptApi
import re



def extract_video_id(url):
    video_id_match = re.search(r'(?:youtu\.be/|youtube\.com/watch\?v=|/embed/|/v/|/e/|watch\?v=|youtu\.be/|embed/|v=)([^&#?]+)', url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        return None

youtube_url = "https://www.youtube.com/watch?v=q-_ezD9Swz4"

video_id = extract_video_id(youtube_url)

data=[]
 
if video_id:
    try:
        srt = YouTubeTranscriptApi.get_transcript(video_id , languages=['en-GB'])
    except:
        srt = YouTubeTranscriptApi.get_transcript(video_id , languages=['en'])


concatenated_data = []
chunk_size = 10  # Adjust as needed
for i in range(0, len(srt), chunk_size):
    chunk = ' '.join(entry['text'] for entry in srt[i:i + chunk_size])  # Concatenate 3 consecutive entries
    concatenated_data.append({'text': chunk})

# Step 2: Create dataset from the concatenated data
formatted_data = [{'page_content': entry['text']} for entry in concatenated_data]
dataset = Dataset.from_dict({"page_content": [entry['page_content'] for entry in formatted_data]})

# Step 3: Convert dataset to documents
data = [{'text': entry['page_content']} for entry in dataset]
docs = [Document(page_content=entry['text']) for entry in data]

# Step 4: Use text splitter to break large chunks into smaller ones
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(docs)

# Print the resulting documents
#print(docs)


# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)


db = Chroma.from_documents(docs,embeddings)
#question = "what is linked list"
#searchDocs = db.similarity_search(question)
#print(searchDocs[0].page_content)





# Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline

retriever = db.as_retriever(search_kwargs={"k": 4})


question = "is it really important for coding and if it is then explain why "



prompt = """
You are an AI assistant named YT Tutor. Use the provided context to answer the question in detail. Provide examples from the context if you cant find any example from the context then answer from your own. Ensure clarity and depth.
Context:
"""

# Set up the pipeline with Qwen-1.5-4B-Chat
pipe = pipeline("text-generation", model="Qwen/Qwen1.5-0.5B")  # Use 'device=0' for GPU

# Create a retriever object from the 'db' with a search configuration where it retrieves up to 4 relevant splits/documents.
context = retriever.get_relevant_documents(question)

# Combine the retrieved context with the prompt
context_text = "\n".join([doc.page_content for doc in context])
full_prompt = prompt + "\n" + context_text + f"\n\nQuestion: {question}\nAnswer:"

# Use the pipeline to generate a detailed answer
result = pipe(full_prompt, temperature=0.7, do_sample=True)

# Extract and print the generated answer
answer = result[0]['generated_text'].split("Answer:")[1].strip()
print(answer)
