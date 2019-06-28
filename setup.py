#!/usr/bin/env python
try:
    from setuptools import setup
except ImportError:
    from distuils.core import setup


setup(
    name='torchpipe',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description='Simple ETL Pipeline for PyTorch',
    long_description=open('./README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yasufumy/torchpipe',
    author='Yasufumi Taniguchi',
    author_email='yasufumi.taniguchi@gmail.com',
    packages=[
        'torchpipe'
    ],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=['torch_nightly'],
    dependency_links=[
        'https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html',
        'https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html',
        'https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html',
    ],
)
