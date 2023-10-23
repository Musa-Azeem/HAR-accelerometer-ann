from setuptools import setup, find_packages

setup(
    name='smokingml',
    python_requires='>3.10.0',
    version='0.1.0', 
    description='Library for training neural networks for Delta project',
    url='https://github.com/Musa-Azeem/HAR-smoking-ml/smokingml',
    author='Musa Azeem',
    author_email='mmazeem@email.sc.edu',
    license='',
    packages=find_packages(),
    install_requires=['torch',
                      'numpy',
                      'pandas',
                      'tqdm',
                      'tabulate',
                      "seaborn",
                      'scikit-learn',
                      'matplotlib'
                      ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS'
        'Programming Language :: Python :: 3',
    ],
)
