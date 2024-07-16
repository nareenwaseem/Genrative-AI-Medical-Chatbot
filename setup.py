from setuptools import find_packages, setup

setup(
    name='Genrative-AI-Medical-Chatbot',
    version='0.0.1',
    author='Nareen Waseem',
    author_email='nareenwaseem15@gmail.com',
    install_requires=[
        "ctransformers==0.2.5",
        "sentence-transformers==2.2.2",
        "pinecone-client",
        "langchain==0.0.225",
        "flask",
        "pypdf",
        "python-dotenv"
    ],
    packages=find_packages(),
)
