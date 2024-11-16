# Redo.io AI Assistant

This is a Proof of Concept (PoC) to leverage LLMs to generate filtering conditions from a natural language user query.

## Dependencies
1. This project requires `Poetry` that can be installed from https://python-poetry.org/docs/#installation
2. Post successful clone of the project on your local, create `.env` file in the root directory of the project and specify these enviroment variable with their repectives values: `OPENAI_API_KEY` (your OpenAI API key)

### Run
1. Run `poetry update` to first install all the dependecies.<br>
2. Run `poetry run chainlit run redo_assistant.py`, where `redo_assistant.py` is the entrypoint of the project.<br>

## Test 
Input:<br>
Load the `demographics.csv` file from the `prison_pop` repository. Sample queries could be:<br>
(a). "Find all individuals who are of the Black ethnicity, sentenced to over 15 years and served over 10 years"<br>
(b). "Find all individuals who are of the Hispanic ethnicity, sentenced to over 10 years, and incarcerated in San Quentin State Prison"<br>

Output:<br>
(a). A .json file with a Pydantic class structure mapping input data columns to their selected values<br>
(b). A .csv file with the selected rows from the input data<br>

## Caution 
Please note that this is an experimental capability in development and has not been released for production use. Tool outputs should not, by any means, be interpreted as an eligible cohort for resentencing.
