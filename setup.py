from setuptools import setup, find_packages


setup(
    name='transformers4RecKDD',
    version='1.0',
    description='Package with tools to train transformers4Rec on kdd Amazon cup data',
    author='Arsenii Vanunts',
    author_email='avanunts@yandex.ru',
    url='https://github.com/avanunts/transformers4RecKDD',
    packages=find_packages(),
    package_data={'paths': ['*.json']},
)
