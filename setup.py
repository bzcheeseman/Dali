import os
from setuptools import setup, find_packages

def readfile(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='recurrent-js-python',
    version='0.0.1',
    description="Python implementation of Andrej Kaparthy's recurrentjs library",
    long_description=readfile('README.md'),
    ext_modules=[],
    packages=find_packages(),
    py_modules = [],
    author='Jonathan Raiman',
    author_email='jraiman at mit dot edu',
    url='https://github.com/JonathanRaiman/recurrentjs',
    download_url='https://github.com/JonathanRaiman/recurrentjs',
    keywords='Gradient Descent, LSTM, neural networks, automatic differentiation',
    license='MIT',
    platforms='any',
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    setup_requires = [],
    install_requires=[
        'numpy'
    ],
    include_package_data=True,
)