# Redo LLM Dev

This is a Proof of Concept (PoC) to leverage LLM to generate filtering conditions from user query.

### Dependencies
1. This project requires `Poetry` that can be installed from https://python-poetry.org/docs/#installation
2. Post successful clone of the project on your local, create `.env` file in the root directory of the project and specify these enviroment variable with their repectives values -
    a. `OPENAI_API_KEY` - your openai api key

### How to run the project ?
1. Run `poetry update` to first install all the dependecies.
2. Run `poetry run python -m app`, where `app.py` is the entrypoint of the project.
