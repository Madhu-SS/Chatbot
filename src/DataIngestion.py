from langchain.document_loaders.url import UnstructuredURLLoader

URLs=[
    "https://www.potentiam.co.uk/"
]

def DataLoader():
    loader=UnstructuredURLLoader(urls=URLs)
    data=loader.load()
    return data