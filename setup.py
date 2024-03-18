"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
Versioning
https://pypi.org/project/setuptools-scm/
"""

from setuptools import setup, find_packages

setup(
    name='model_poc_detector',
    package_dir={'': '.'},
    packages=find_packages(),
    python_requires='>=3.9, <4',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    package_data={
        # include any txt files in Data
        '': ['data/*.txt']
    },
    #scripts=[''],
    install_requires=[
        'numpy>=1.24',
        'ai-model-utils @ git+ssh://git@github.com/humanLearning/ai-model-utils@v5.10.5',
        'torch>=2.0.0', 'torchaudio>=2.0.1', 'torchvision>=0.15.1'
    ],
    extras_require={}
)
