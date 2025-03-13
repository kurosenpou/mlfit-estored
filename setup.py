# setup.pyの正しい書き方（entry_pointsが必須です！）
from setuptools import setup, find_packages

setup(
    name='mlfit',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'tensorflow',
        'scipy',
        'scikit-learn',
        'xgboost',
        'lightgbm',
        'catboost'
    ],
    entry_points={
        'console_scripts': [
            'mlfit=main:main',  # ←必須
        ],
    },
)
