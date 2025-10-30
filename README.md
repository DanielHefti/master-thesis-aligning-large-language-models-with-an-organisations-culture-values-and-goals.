

# Aligning Large Language Models with an Organization's Culture, Values and Goals
This repository contains the code, data, and analysis for the master thesis "Aligning Large Language Models with an Organization's Culture, Values and Goals" by Daniel Pascal Hefti. Please note that, due to the participating organizations' privacy concerns, only the experimental data from a publicly available dataset (the public phase of the master's thesis) is available.

## Repository Structure

- `dataset/`  
	- Contains public datasets used in experiments.
- `experiment/`  
	Contains scripts and data for running experiments, evaluating model responses, and sampling.
    - `analysis/` - Contains statistical analysis scripts and results.
	- `evaluation` - Contains CSV files with grading from evaluator model.
	- `responses/` — Raw and joined model response CSVs, organized by phase.  
	- `sampling/` — Scripts for selecting models and responses for human evaluation.  

## Important files
- **experiment\public_dataset_full-prompt.txt** - Prompt issued to LLMs including instruction for treatment group.
- **experiment\public_dataset_without_instruction-prompt.txt** - Prompt issued to LLMs without instruction for control group.
- **experiment\reference_answer.md** - Reference answer provided to evaluator LLM.
- **experiment\score_rubrics_en.json** - List of score rubrics used for evaluation.


## Main Scripts
- **experiment\experiment.py** - Runs experiment.
- **experiment\evaluation_with_reference_answer.py** - Evaluates responses from experiment with reference answer.
- **experiment\evaluation_without_reference_answer.py** - Evaluates responses from experiment without reference answer.
- **experiment\join_responses.py** - Joins all responses to facilitate downstream analysis.
- **experiment\analysis\with_reference_answer\generate_stats.py** - Analyzes and synthesizes results from experiment with reference answer.
- **experiment\analysis\without_reference_answer\generate_stats.py** - Analyzes and synthesizes results from experiment without reference answer.

## Getting Started

1. Clone the repository.
2. Install dependencies:
	 ```
	 pip install -r requirements.txt
	 ```
3. Run the scripts in the `experiment/` folder as needed, following the workflow described in the thesis.


## Project Purpose

The goal of this project is to evaluate and align large language models with organizational culture, values, and goals. It provides tools for automatic and human evaluation, statistical analysis, and reproducible experiments.
