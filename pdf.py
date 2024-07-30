import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage

def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=index_name)
        index = load_index_from_storage(storage_context)
    return index

pdf_path = os.path.join("data", "pdf")
countries_pdf = SimpleDirectoryReader(pdf_path).load_data()
countries_index = get_index(countries_pdf, "countries_pdf")
countries_engine = countries_index.as_query_engine()