from setuptools import setup, find_packages

setup(
    name='mix_eval',
    version='1.0.0',
    author='MixEval team',
    author_email='jinjieni@nus.edu.sg',
    packages=find_packages(),
    install_requires=[
        'transformers>=4.41.0',
        'tiktoken>=0.6.0',
        'fschat>=0.2.36',
        'SentencePiece>=0.2.0',
        'accelerate>=0.30.1',
        'pandas>=2.2.2',
        'scikit-learn>=1.5.0',
        'hf_transfer>=0.1.6',
        'openai>=1.30.5',
        'httpx',
        'nltk',
        'numpy',
        'tqdm',
        'protobuf',
        'python-dotenv',
        'anthropic',
        'mistralai',
        'google-generativeai',
        'google-cloud-aiplatform',
        'reka-api',
        'dashscope',
        'fastapi >= 0.114.1', # To avoid this bug https://github.com/vllm-project/vllm/issues/8212
        'pydantic >= 2.9.0'
    ],
    package_data={
    },
    entry_points={
    },
    url='https://mixeval.github.io/',
    license='License :: OSI Approved :: MIT License',
    description='A state-of-the-art benchmark and eval suite for Large Language Models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
