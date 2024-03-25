from setuptools import find_packages,setup

setup(
    name='chatbot',
    version='0.0.1',
    author='madhu s s',
    author_email='madhugowda426@gmail.com',
    install_requires=['libmagic','unstructured','pypdf','langchain','ctransformers','streamlit','sentence_transformers','huggingface_hub','faiss_cpu'],
    packages=find_packages()
)