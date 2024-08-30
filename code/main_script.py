import subprocess
import os

# List of script filenames to be run
scripts = [
    "discrete_noise_script_1.py",
    "discrete_noise_script_2.py",
    "weak_noise_discrete_script_1.py",
    "weak_noise_discrete_script_2.py",
    "weak_noise_discretisable_script_1.py",
    "weak_noise_discretisable_script_2.py",
]

# Execute each script
for script in scripts:
    script_path = os.path.join(os.path.dirname(__file__), script)
    print(f"Running {script}...")
    try:
        subprocess.run(["python3", script_path], check=True)
        print(f"Finished running {script}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}\nContinuing with next script...\n")

print("All scripts have been executed.")