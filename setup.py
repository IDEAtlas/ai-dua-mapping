import argparse
import os
import subprocess
import sys
import platform

def execute_command(command_list):
    print(f"Executing: {' '.join(command_list)}")
    try:
        subprocess.run(command_list, check=True)
    except subprocess.CalledProcessError as error:
        print(f"Process failed with exit code {error.returncode}")
        sys.exit(error.returncode)

def setup_conda():
    execute_command(["conda", "config", "--add", "channels", "conda-forge"])
    execute_command(["conda", "config", "--set", "channel_priority", "flexible"])
    execute_command(["conda", "env", "update", "-n", "ideatlas", "-f", "./env/conda.yaml", "--prune", "--solver=libmamba"])

def get_docker_run_command():
    host_path = os.path.abspath("./")
    cmd = ["docker", "run", "-dit", "--name", "ideatlas"]
    cmd.extend(["-p", "8888:8888"])

    if platform.system() != "Darwin":
        cmd.extend(["--gpus", "all"])

    # Run container with current user permissions to avoid file permission issues on Linux
    # if platform.system() == "Linux":
    #     cmd.extend(["-u", f"{os.getuid()}:{os.getgid()}"]) 

    cmd.extend(["-v", f"{host_path}:/ai-dua-mapping", "ideatlas"])
    
    return cmd

def setup_docker():

    execute_command(["docker", "build", "-t", "ideatlas", "./env/"])
    execute_command(get_docker_run_command())
    # execute_command(["docker", "exec", "-it", "ideatlas", "bash"])

def main():
    parser = argparse.ArgumentParser(description="Environment initialization.")
    parser.add_argument("--conda", action="store_true")
    parser.add_argument("--docker", action="store_true")
    
    args = parser.parse_args()

    run_conda = args.conda or not args.docker
    run_docker = args.docker

    if run_conda:
        setup_conda()

    if run_docker:
        setup_docker()

if __name__ == "__main__":
    main()