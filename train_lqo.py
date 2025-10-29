import os

import click

from src.optimizers.xgboost_lqo import XGBoostLQO


@click.command()
@click.option(
    "--data_file",
    default="results/job_training_data_cache.parquet",
    help="Path to the training data .parquet file.",
)
def main(data_file):
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    print(f"Starting training process using data from {data_file}")

    lqo = XGBoostLQO()
    lqo.train(training_data_path=data_file)

    print("\n--- Training successfully completed! ---")
    print(f"Model saved to: {lqo.model_path}")
    print(f"Featurizer saved to: {lqo.featurizer_path}")


if __name__ == "__main__":
    main()
