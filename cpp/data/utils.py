import os
import subprocess

def print_progress(fraction_completed, total_work=1.0):
    progress = fraction_completed/total_work
    print("█" * (int(20 * (progress))) + " %.1f%% \r" % (100 * progress,), end="", flush=True)


def execute_bash(command):
    """Executes bash command, prints output and throws an exception on failure."""
    #print(subprocess.check_output(command.split(' '), shell=True))
    process = subprocess.Popen(command,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)
    for line in process.stdout:
        print(line, end='', flush=True)
    process.wait()
    assert process.returncode == 0
