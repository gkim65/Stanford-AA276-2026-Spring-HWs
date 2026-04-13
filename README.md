# Stanford AA276: Principles of Safety-Critical Autonomy, Spring 2026

## Overview

This repository contains the coding portion of the homework assignments for AA 276: Principles of Safety-Critical Autonomy, Spring 2026.<br>
To get started on a specific homework, see the homework handout on Canvas, as well as the `README.md` in the respective subfolder here, e.g., `./hw1/`.

For any questions, please reach out to the course staff on Ed or via email.

## Recommended Workflow

Some steps in the homeworks will require access to a GPU, e.g., for training a neural network model.<br>
For these steps, we will provide cloud credits for you to use in the Google Compute Engine, if you do not already have access to a GPU.<br>
However, it can be difficult and costly to develop directly on a virtual machine in the Google Compute Engine.<br>
Keeping that in mind, we highly recommend that you use the following workflow:

**For code development**, we suggest that you implement your code locally using your preferred IDE<br>
and run the provided **test scripts** directly on your computer, which should work even without a GPU.<br>
By doing so, you should be able to finish the majority of the coding assignment without issue.<br>
You might find it useful to use Git both for version control and for the future steps that require you to move<br>
your code to virtual machines in the Google Compute Engine.

**Once you have passed the provided test scripts**, you can start a virtual machine in the Google Compute Engine and<br>
run the scripts that require a GPU, which will be marked in the homework as **GPU-based scripts**.<br>
These scripts may produce data that will be used by future scripts or should be submitted as part of the assignment,<br>
so you should try to run all **GPU-based scripts** on the same machine, or remember to move the data appropriately.<br>
Such outputs will be indicated clearly in the homework instructions.<br>
**!!! After you finish running a GPU-based script, you should shutdown the virtual machine to avoid accumulating fees in Google Compute Engine.**

## Virtual Environment Setup

For convenience, we strongly encourage you to maintain a consistent Python environment with the virtual environment manager<br>
`python3-venv` and the package manager `python3-pip` to use across all homeworks. In each homework `README.md`, we will<br>
specify the Python packages that you need in order to run the **test scripts**. We will also provide instructions on how to set up your<br>
a Conda virtual environment in the Google Compute Engine virtual machine, to run the **GPU-based scripts**.

Here are some (Linux) commands you might find useful:<br>
`sudo apt install [package_name]` - install an OS package.<br>
`python -m venv env` - create a virtual Python environment `env` in the current directory.<br>
`source env/bin/activate` - activate the virtual Python environment. Subsequently, packages installed with `pip` will be kept within this environment.<br>
`deactivate` - deactivate the virtual Python environment.<br>

Please refer to the respective homework `README.md` for what packages need to be installed.

If you encounter any environment issues, please reach out to us on Ed or via email.

## Spawning Virtual Machines on Google Compute Engine

Please see [this tutorial](https://docs.google.com/document/d/1btNdbHUv_ErEz2ccBMA5xL8scMc3octeCv9BheKb0wQ/edit?usp=sharing) to get started with spawning virtual machines on Google Compute Engine.<br>
It is important to keep an eye on your credit usage to avoid being charged!<br>
Shutdown your virtual machines when not in use and set budget alerts!

**!!! You should run training scripts within `tmux` sessions, so that they are not killed if your `ssh` disconnects.**<br>
`tmux` is a virtual session manager. You might find the following commands useful:<br>
`tmux ls` - list existing tmux sessions.<br>
`tmux new -s [name]` - create a new tmux session.<br>
`tmux a -t [name]` - attach to an existing tmux session. Within a tmux session, detach with: `Ctrl+b`, and then `d`. Exit as you normally would with `exit`.<br>
