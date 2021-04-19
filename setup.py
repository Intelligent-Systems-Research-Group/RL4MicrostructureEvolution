from setuptools import setup

setup(
    name='msevo_gym',
    version='0.1',
    keywords='rl, environment, openaigym, openai-gym, gym, optimal control',
    author="Johannes Dornheim",
    author_email='johannes.dornheim@mailbox.org',
    install_requires=[
        'gym>=0.12',
        'matplotlib',
        'pyquaternion',
        'imageio',
        'pymks',
        'scipy',
        'numpy'
    ],
    packages=["msevolution_env",
              "msevolution_env.envs"],
    include_package_data=True
)

"""
        'pandas>=0.23',
        'numpy>=1.15',
        'scipy>=1.1',
        'imageio >= 2',
        'scikit-learn >= 0.20',
        'matplotlib',
        'pymks',
        'pyquaternion'
"""