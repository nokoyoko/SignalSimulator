# SignalSimulator

A repository for code related to a project about signal. You can read a short synopsis of the project and it's goals [here](https://arxiv.org/pdf/2305.09799). It has now been accepted for publication -- the link will be provided when it is available in full. 

## What's Inside 
The files are as follows:
* message generation to model communication in a social network: [`rfmp_msg_gen.py`](https://github.com/nokoyoko/SignalSimulator/blob/main/rfmp_msg_gen.py)
* a standard attack script, to be modified for different settings: [`updated_attack.py`](https://github.com/nokoyoko/SignalSimulator/blob/main/updated_attack.py)
* a script to generate different social graphs for all testing environments: [`graph_gen_master_realistic.py`](https://github.com/nokoyoko/SignalSimulator/blob/main/graph_gen_master_realistic.py)
* group member files to create realistic groups: [`group_member_counts.tsv`](https://github.com/nokoyoko/SignalSimulator/blob/main/group_member_counts.tsv) -- the paper which this data was obtained from may be found [here](https://gvrkiran.github.io/content/whatsapp.pdf)

## How to Use It
Prior to working with the code, the social graphs need to be made. This may be done by the following command:

```bash
    python3 graph_gen_master_realistic.py
```

The following command is an example to call the code for a particular experiment:

```bash
    nohup python3 -u rfmp_msg_gen.py --env default --graph gMartiny --exp standard --atk updated --mar true --defs self --str 5 > output_martiny_def_str5.txt &
```

Relevant Syntax:
* `--env` : the desired environment e.g. standard or super user(s)
* `--graph` : the social graph to execute on e.g. gMartiny, gLarge, gGeneral
* `--exp` : the experiment being run, e.g. Standard, Large
* `--atk` : the attack file (always use updated)
* `--mar` : enable Martiny style generation/attack (not required)
* `--defs` : enable the defense or not (not required, only defense is 'self') 
* `--str` : strength of the defense

## Some Notes
The variables `runs` and `numEpochs` are hard coded in the `rfmp_msg_gen.py` file. They determine the following:
* `runs` : the number of experiments executed in parallel. Each will be we run on a separate CPU core, so keep that in mind. 
* `numEpochs` : the number of epochs to generate in each attack. 

This program can consume *a lot* of memory. For example, at 20 runs and 9000 epochs the code will consume upwards of 220G of memory. 

## What's Coming Next
There are no concrete plans to advance the project as of now. More sophisticated defenses would be interesting to explore. If you are interested in collaborating please feel free to contact me. 

## Contact
eric (dot) brigham4 (at) gmail.com