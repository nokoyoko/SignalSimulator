# SignalSimulator

A repository for code related to a project about signal. You can read a short synopsis of the project and it's goals [here](https://arxiv.org/pdf/2305.09799).

## What's Inside 
At the time of writing, the document contains a few files:
    * message generation to model communication in a social network: [`rfmp_msg_gen.py`](#https://github.com/nokoyoko/SignalSimulator/blob/main/rfmp_msg_gen.py)
    * a standard attack script, to be modified for different settings: [`notime_cor_attack_variable_flurries_standard.py`](#notime_cor_attack_variable_flurries_standard)
    * a script to generate different social graphs for all testing environments: [`graph_gen_master_realistic.py`](#https://github.com/nokoyoko/SignalSimulator/blob/main/graph_gen_master_realistic.py)
    * other files containing data necessary to support the above

## What's Coming Next
The project will be advanced in a number of ways:
    * Like the generation script, the attack script will be modified to be modular -- that is, it will incorporate all of the generation environments. It's current iteration only supports an attack on the standard messaging environment. While these attacks already exist, they are in separate files. Their logic will be merged into a single attack script. 
    * Graphical results of the attacks will be added as experiments are performed. 
    * Links to longer versions of the paper will be added as they come out. 

## Contact
eric (dot) brigham4 (at) gmail.com