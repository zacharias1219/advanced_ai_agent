from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from dotenv import load_dotenv
from llama_index.core.tools import CodeReaderTool

load_dotenv()

llm = Ollama(model = 'mistral', request_timeout = 30.0)

parser = LlamaParse(result_type = 'markdown')
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()
code_reader = CodeReaderTool()


emebd_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(embed_model=emebd_model, documents=documents)
query_engine = vector_index.query_engine(llm=llm)

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="this gives documentation about code for an API. Use this for reading docs for the API",
        ),
    ),
    code_reader,
]