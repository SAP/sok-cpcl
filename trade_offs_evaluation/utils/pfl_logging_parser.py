"""
PFL Logging Parser Utilities

This module provides utilities for parsing and aggregating training metrics
from the PFL (Private Federated Learning) library. It extracts the best
validation statistics from CSV logs and saves them in a JSON file.
"""

import pandas as pd
import json


def save_best_stats_per_iteration(
	hyperparameters,
	iteration,
	dataset_name,
	base_folder_metrics,
	base_path_metrics
):
	"""Extract and save the best validation statistics from a training iteration.
	
	The PFL library outputs training statistics (accuracy, loss) to CSV files
	during each iteration. This function reads the CSV, identifies the best
	validation accuracy achieved, and aggregates results across all iterations
	into a consolidated JSON file for comparison and analysis.
	
	Args:
		hyperparameters (dict): Configuration parameters used in this training iteration.
			Keys typically include 'noise_multiplier', 'epsilon', 'batch_size', etc.
		iteration (str or int): Unique identifier for this training run/iteration.
		dataset_name (str): Name of the dataset (e.g., 'mnist', 'cifar10').
			Used to organize results by dataset in the output JSON.
		base_folder_metrics (str): Directory where the aggregated JSON results will be stored.
			Example: './results'
		base_path_metrics (str): Path prefix for the CSV metrics file (without extension).
			The function will read: '{base_path_metrics}_{iteration}.csv'
			Example: './logs/metrics' → reads './logs/metrics_1.csv'
	
	Returns:
		None. Writes/appends results to JSON file at:
		'{base_folder_metrics}/best_stats_{dataset_name}.json'
	
	Output JSON Format:
		{
			"<iteration>": {
				"accuracy": <float>,           # Best validation accuracy achieved
				"loss": <float>,               # Validation loss at best accuracy
				"best_accuracy_iteration": <int>,  # Central iteration where best accuracy occurred
				"hyperparameters": <dict>      # Configuration used for this run
			},
			...
		}
	
	Example:
		>>> save_best_stats_per_iteration(
		...     hyperparameters={'noise_multiplier': 1.5, 'epsilon': 1.0},
		...     iteration='1',
		...     dataset_name='mnist',
		...     base_folder_metrics='./results',
		...     base_path_metrics='./logs/metrics'
		... )
		Iteration: 1 - Best accuracy: 0.9523 - Best loss: 0.1234 - Best accuracy iteration: 42
	
	Raises:
		FileNotFoundError: If the CSV metrics file does not exist.
		KeyError: If expected columns are missing from the CSV.
	"""

	# ✓ Construct output path for aggregated JSON results
	json_best_stats_path = f"{base_folder_metrics}/best_stats_{dataset_name}.json"
	pfl_metrics = pd.read_csv(f"{base_path_metrics}_{iteration}.csv")
	pfl_metrics = pfl_metrics.dropna(subset=['Central val | accuracy'])


	# fl_metrics['Central val | accuracy'] , central_iteration, Central val | loss, Central val | accuracy
	# Take the row with the highest 'Central val | accuracy'
	# save the iteration, the accuracy and the loss
	best_accuracy = pfl_metrics['Central val | accuracy'].max()
	best_accuracy_row = pfl_metrics[pfl_metrics['Central val | accuracy'] == best_accuracy]
	best_accuracy_row = best_accuracy_row.iloc[0]
	best_accuracy_iteration = best_accuracy_row['central_iteration']
	best_accuracy_loss = best_accuracy_row['Central val | loss']


	# Open the json file, and append the new data with {iteration: {'accuracy': accuracy, 'loss': loss, 'hyperparameters': hyperparameters}}
	# If the file does not exist, create it
	# If it's empty, create the first entry

	try:
		with open(f"{json_best_stats_path}", 'r') as f:
			best_stats = json.load(f)
	except:
		best_stats = {}

	best_stats[iteration] = {
		'accuracy': best_accuracy,
		'loss': best_accuracy_loss,
		'best_accuracy_iteration': best_accuracy_iteration,
		'hyperparameters': hyperparameters
	}

	with open(f"{json_best_stats_path}", 'w') as f:
		json.dump(best_stats, f)

	print(f"Iteration: {iteration} - Best accuracy: {best_accuracy} - Best loss: {best_accuracy_loss} - Best accuracy iteration: {best_accuracy_iteration}")
