#!/usr/bin/env python
try:
    from setuptools import setup
except ImportError:
    from distuils.core import setup


setup(
    name='pytorch-pipeline',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description='Simple ETL Pipeline for PyTorch',
    long_description=open('./README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yasufumy/pytorch-pipeline',
    author='Yasufumi Taniguchi',
    author_email='yasufumi.taniguchi@gmail.com',
    packages=[
        'pytorch_pipeline'
    ],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=['torch>=1.2.0', 'numpy', 'lineflow'],
)
