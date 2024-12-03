import argparse
import os
import yaml
from pathlib import Path
from mirgecom.io import read_and_distribute_yaml_data

def parse_range(value):
    """Parse the Range(begin, end, step) string and return a list of values."""
    if value.startswith("Range(") and value.endswith(")"):
        try:
            args = value[6:-1].split(",")
            begin, end, step = map(float, args)
            return [begin + i * step for i in range(int((end - begin) / step) + 1)]
        except Exception as e:
            raise ValueError(f"Invalid Range format: {value}") from e
    return None


def generate_input_files(template_file, output_dir, casename):
    """Generate input files based on Range values in the template YAML."""
    data = read_and_distribute_yaml_data(None, template_file)
    output_files = []
    for key, value in data.items():
        if isinstance(value, str):
            range_values = parse_range(value)
            if range_values:
                # Generate multiple input files
                for v in range_values:
                    modified_data = data.copy()
                    modified_data[key] = v
                    filename = f"{casename}_{key}_{v:.3g}.yaml"
                    output_file = Path(output_dir) / filename
                    with open(output_file, "w") as out_f:
                        yaml.safe_dump(modified_data, out_f)
                    output_files.append((filename, key, v))

    return output_files


def generate_bash_script(input_files, script, casename, output_dir):
    """Generate a bash script to run the simulation with each input file."""
    bash_script_path = Path(output_dir) / f"{casename}_multi_bsub.sh"
    with open(bash_script_path, "w") as bash_script:
        bash_script.write("#!/bin/bash\n\n"
                          "#BSUB -nnodes 1\n"
                          "#BSUB -G uiuc\n"
                          "#BSUB -W 300\n"
                          f"#BSUB -J {casename}\n"
                          "#BSUB -q pbatch\n"
                          f"#BSUB -o {casename}.txt\n\n"
                          "source ../emirge/config/activate_env.sh\n"
                          "source ../emirge/mirgecom/scripts/mirge-testing-env.sh\n\n")
        for input_file, param_name, param_value in input_files:
            output_filename = f"{casename}_{param_name}_{param_value:.3g}_out.txt"
            base_cmd = f"$MIRGE_MPI_EXEC -n 1 $MIRGE_PARALLEL_SPAWNER nvprof python -m mpi4py"
            cmd_options = f"{script} -i {input_file} --lazy >& {output_filename}"
            cmd = f"{base_cmd} {cmd_options}"
            bash_script.write(f"{cmd}\n")
    os.chmod(bash_script_path, 0o755)  # Make the script executable
    print(f"Bash script generated: {bash_script_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate input files and a bash script for simulations.")
    parser.add_argument("template_file", type=str, help="Path to the YAML template input file.")
    args = parser.parse_args()

    template_file = Path(args.template_file)
    if not template_file.exists():
        raise FileNotFoundError(f"Template file {template_file} not found.")

    # Derive the casename from the file root
    casename = template_file.stem
    output_dir = Path(f"generated_inputs_{casename}")
    script = "./driver.py"  # Simulation script

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate input files
    print("Generating input files...")
    input_files = generate_input_files(template_file, output_dir, casename)
    print(f"Generated {len(input_files)} input files.")

    # Step 2: Generate bash script
    print("Generating bash script...")
    generate_bash_script(input_files, script, casename, output_dir)


if __name__ == "__main__":
    main()
