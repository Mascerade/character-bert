from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='character-bert',
   version='1.0',
   description='CharacterBERT: Reconciling ELMo and BERT for Word-Level Open-Vocabulary Representations From Characters',
   license="Apache License 2.0",
   long_description=long_description,
   author='Hicham El Boukkouri',
   author_email='helboukkouri.dev@gmail.com',
   url="https://github.com/Mascerade/character-bert",
   packages=['character_bert'],
   install_requires=['torch',
                     'numpy',
                     'sklearn',
                     'transformers']
)