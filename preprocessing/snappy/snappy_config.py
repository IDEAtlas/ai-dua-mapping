import psutil
import os
import subprocess

def update_java_max_mem():
    """
    Updates the java_max_mem value in snappy.ini dynamically based on available memory.
    """
    # Calculate 80% of total system memory in GB
    java_max_mem = int(0.8 * psutil.virtual_memory().total / 1024**3)
    print(f"Calculated java_max_mem: {java_max_mem}G")

    # Path to snappy.ini
    snappy_ini_path = os.path.expanduser("~/.snap/snap-python/snappy.ini")
    if not os.path.exists(snappy_ini_path):
        raise FileNotFoundError(f"{snappy_ini_path} does not exist. Ensure the SNAP Python bindings are initialized.")

    # Update the java_max_mem value using sed
    try:
        subprocess.run(
            f"sed -i 's/java_max_mem: .*G/java_max_mem: {java_max_mem}G/' {snappy_ini_path}",
            shell=True,
            check=True,
        )
        print(f"Updated java_max_mem to {java_max_mem}G in {snappy_ini_path}.")
    except subprocess.CalledProcessError as e:
        print(f"sed failed: {e}. Attempting manual update.")
        # Fallback: Update snappy.ini with Python
        with open(snappy_ini_path, "r") as file:
            lines = file.readlines()
        updated_lines = [f"java_max_mem: {java_max_mem}G\n" if "java_max_mem:" in line else line for line in lines]
        with open(snappy_ini_path, "w") as file:
            file.writelines(updated_lines)
        print(f"Manually updated java_max_mem to {java_max_mem}G in {snappy_ini_path}.")