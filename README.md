# Continual Pre-Training for Small Language Models


## About
Blablabla


## Usage
### Requirements
Please make sure to have the following installed within a linux environment:
- Uv (follow installation instructions from https://docs.astral.sh/uv/getting-started/installation/)
- Python (== 3.12.12)
- Cuda (== 12.9)


### Installing the dependencies
1. Compile the uv lock file to a requiements file:
`uv pip compile pyproject.toml -o requirements.txt`
2. Create a virtual environment: 
`uv venv <path-to-your-virtual-environment> --python 3.12.12`
3. Activate your virtual environment:
`source <path-to-your-virtual-environment>/bin/activate`
4. Install the dependencies:
`uv add -r requirements.txt`
5. Install the project
`uv pip install -e .`


### Running the functionalities
#### Data collection
#### Model continual pretraining
#### Model evaluation


