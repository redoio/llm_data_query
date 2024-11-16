# Redo.io AI Assistant

This is a Proof of Concept (PoC) to leverage LLM to generate filtering conditions from a natural language user query.

### Dependencies
1. This project requires `Poetry` that can be installed from https://python-poetry.org/docs/#installation
2. Post successful clone of the project on your local, create `.env` file in the root directory of the project and specify these enviroment variable with their repectives values -
    a. `OPENAI_API_KEY` - your OpenAI api key

### Run
1. Run `poetry update` to first install all the dependecies.
2. Run `poetry run chainlit run redo_assistant.py`, where `redo_assistant.py` is the entrypoint of the project.

### Caution 
Please note that this capability is in development and has not been released for production use. 
