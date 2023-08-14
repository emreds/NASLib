import subprocess

import yaml

from naslib import utils


def update_yaml_value(yaml_file: str, keys: list, values: list) -> None:
    # Read the YAML file and load its contents into a dictionary
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    # Update the value in the dictionary
    for key, value in zip(keys, values):
        data[key] = value

    # Write the updated dictionary back to the YAML file
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file)

def trigger_job(command_to_run: str) -> None:
    try:
        # Run the command and capture the output and error streams
        result = subprocess.run(command_to_run, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Print the output and error (if any)
        print("Output:")
        print(result.stdout)
        print(result.stderr)

        # Print the return code of the command
        print("\nReturn code:", result.returncode)

    except subprocess.CalledProcessError as e:
        print("Error occurred:", e)


if __name__ == "__main__":
    yaml_file_path = "/p/project/hai_nasb_eo/emre/data_centric/NASLib/naslib/runners/predictors/new_predictor.yaml"
    keys_to_update = ["model_metric", "seed", "train_size_single"]
    command_to_run = f"python runner.py --config-file {yaml_file_path}"

    for model_metric in ["val_accuracy", "train_time"]:
        for seed in [17, 21, 42, 81, 123]:
            for train_size in [300, 400, 500, 600, 700, 800, 900, 1000]:
                values = [model_metric, seed, train_size]
                update_yaml_value(yaml_file_path, keys_to_update, values)
                print(f"Running now with \n model_metric: {model_metric} \n seed: {seed} \n train_size: {train_size} \n")
                trigger_job(command_to_run)
        
    print("DONE!")
