from setuptools import setup, Command
from codecs import open
from os import path

currPath = path.abspath(path.dirname(__file__))

# Parse README
with open(path.join(currPath, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Parse version
with open(path.join(currPath, 'snapshot_ensemble', '__init__.py')) as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('"')[1]

setup(
    name='snapshot_ensemble',
    description='Train TensorFlow Keras models with Cosine Annealing and save an ensemble of models with no additional computational expense.',
    version=version,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/adamvvu/snapshot_ensemble',
    author='Adam Wu',
    author_email='adamwu1@outlook.com',
    packages=['snapshot_ensemble'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Programming Language :: Python',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'tensorflow',
	'matplotlib',
	'glob',
    ],
    license_files = ('LICENSE',),
)