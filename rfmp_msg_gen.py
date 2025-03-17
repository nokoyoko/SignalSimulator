from ipaddress import collapse_addresses
from operator import truediv
from pickle import TRUE
from plistlib import UID
from igraph import *
import json
import attack as atk
import cor_attack as cor_atk
# attack imports TODO: improve this -- stretch goal
import notime_cor_attack_variable_flurries_standard
import notime_cor_attack_variable_flurries_large
import notime_cor_attack_variable_flurries_popular
import notime_cor_attack_variable_flurries_multi_popular_group_user
import notime_cor_attack_variable_flurries_with
import notime_cor_attack_variable_flurries_with
import notime_cor_attack_variable_flurries_with_popular
import notime_cor_attack_variable_flurries_multi
import martiny_attack_recreate
# end attack imports
import matplotlib.pyplot as plt
import numpy as np
from numpy import where
import random
import pandas as pd
import multiprocessing
from matplotlib.ticker import MaxNLocator, MultipleLocator
from graph_gen_master_realistic import genGroups
import os
import argparse

def LogStandardSort(d):
    return d['time']

def UIDSort(p):
    return p['UID']

ENVIRONMENTS = {
    "default": { "super_users": False},
    "super_users": {"super_users": True, "super_user_ratio": 0.1, "boost_factor": 10, "specific_super_user": None},
    # note that the next environment is for user 0 being a super user in a non-injected group => the next two cases are not symmetric
    "super_user_0": {"super_users": True, "super_user_ratio": 0.1, "boost_factor": 10, "specific_super_user": 0},
    "super_user_group": {"super_users": True, "super_user_ratio": 0.1, "boost_factor": 10, "specific_super_user": None, "group_super_user": True}
}

# command line parsing to grab messaging style (default or super user) and add'l the graph *.tmp
#  --env:   specify if working with standard or super user environments
#  --graph: identify the graph being worked with (omit .tmp)
#  --exp:   identify the experiment for the save file (.png) as well as to grab the particular attack script 
parser = argparse.ArgumentParser(description="Select testing environment from among default/uniform messaging and super users.")
parser.add_argument("--env", choices=["default", "super_users"], default="default", help="Select the simulation environment.")
parser.add_argument("--graph", required=True, help="Select the graph file name e.g. gGeneral.tmp.")
parser.add_argument("--exp", type=str, required=True, help="Experiment identifier e.g. standard, large group, &c.")
parser.add_argument("--atk", type=str, required=True, help="Specify the attack/defense type (e.g., standard, large, &c).")
parser.add_argument("--defs", type=str, required=False, help="Specify the defense, if any.")
parser.add_argument("--str", type=str, required=False, help="Specify the defense strength (1-10)")
args = parser.parse_args()
CURRENT_ENV = args.env
GRAPH_FILE = args.graph
EXPERIMENT = args.exp
ATTACK_TYPE = args.atk
DEF = args.defs
DEF_STR = args.str
CONFIG = ENVIRONMENTS[CURRENT_ENV]


# a function to grab the super_users as needed 
def select_super_users(graph, group_data):
    # returns super user set based on environment
    super_users = set()
    # on 0 is super user
    if CONFIG["specific_super_user"] is not None:
        super_users.add(CONFIG["specific_super_user"])
        # If user 0 is the specific super user, assign them to exactly one other group (not {0,1,2})
        if CONFIG["specific_super_user"] == 0 and CONFIG.get("extra_group_super_user") is True:
            eligible_groups = [group for group in group_data.values() if not set(group).issubset({0,1,2})]
            if eligible_groups:
                extra_group_for_user_0 = random.choice(eligible_groups)
            else:
                extra_group_for_user_0 = {0, 10, 11, 12}
    # on injected group super user
    elif CONFIG["group_super_user"]:
        super_users.add(random.choice([0, 1, 2]))
    # on super_user_ratio*TotalUsers many super users
    elif CONFIG["super_users"]:
        eligible_users = set(range(len(graph.vs))) - {0, 1, 2, 4, 5}
        super_user_count = int(len(graph.vs) * CONFIG["super_user_ratio"])
        super_users.update(random.sample(eligible_users, super_user_count))
    print(f"Super User Set: f{super_users}")
    return super_users

# generate poisson values for later group messaging
def generatePoissonValues(lambda_param, num_entries):
    poisson_values = np.random.poisson(lambda_param, num_entries)
    return poisson_values

seen_uids = set()
def check_duplicate_uid(uid):
    if uid in seen_uids:
        raise ValueError(f"[ERROR] Duplicate UID detected: {uid}")
    seen_uids.add(uid)

def processMessage(sender, receiver, msg, group_data, log, seen_messages, thisUID, poisson_value, rr_log):
    # Ensure message forwarding for entire group
    # Args:
    # > sender (tuple): (sender_id, receiver_id)
    # > msg (dict): message data
    # > group_data (dict): {group_id: [member1, member2, ...]} <-- created by graph_gen_master_realistic.py
    # > log (list): list to store messages
    # > thisUID (int): unique message ID counter
    # > poisson_value (int): number of expected group messages for this epoch

    # Return: (int) updated UID
    
    sender_id = int(sender)
    receiver_id = int(receiver)
    #print(f"sender_id: {sender_id}")
    #print(f"receiver_id: {receiver_id}")

    if poisson_value > 0:
        for group_id, members in group_data.items():
            if sender_id in members and receiver_id in members:
                print(f"\n [FORWARDING] Group {group_id}: Sender {sender_id} -> {receiver_id}")

                group_messages = [] # forwarded group messages
                read_receipts =  [] # read receipts

                # fwd messages to all members xc sender
                for member in members:
                    if member != sender_id:
                        print(f"[DEBUG] Checking UID {msg['UID']} before forwarding {sender_id} -> {receiver_id}")
                        # [NEW] condition to prevent double-sending to receiver_id
                        if member == receiver_id and (msg['time'], sender_id, receiver_id, msg['UID']) in seen_messages:
                            print(f"Skipping duplicate message with ID {msg['UID']}, between {sender_id} -> {receiver_id}")
                            # [NEW] Generate an RR for skipped message
                            thisUID += 1
                            check_duplicate_uid(thisUID)
                            rrMsg = {'time': msg['time'], 'type': 1, 'from': receiver_id, 'to': sender_id, 'UID': thisUID}
                            read_receipts.append(rrMsg)
                            continue

                        # create and store forwarded message
                        new_msg = msg.copy()
                        thisUID += 1
                        check_duplicate_uid(thisUID)
                        new_msg['UID'] = thisUID
                        new_msg['to'] = member
                        message_tuple = (new_msg['time'], new_msg['from'], new_msg['to'], new_msg['UID'])

                        print(f"[DEBUG UID, GROUP] Assigning UID: {thisUID} to message from {new_msg['from']} to {new_msg['to']}")

                        if message_tuple not in seen_messages:
                            group_messages.append(new_msg)
                            seen_messages.add(message_tuple)

                            if new_msg['from'] in {0, 1, 2} and new_msg['to'] in {0, 1, 2}:
                                print(f"[LOG MESSAGE] {new_msg['from']} -> {new_msg['to']} | UID: {new_msg['UID']} | Time: {new_msg['time']}")
                        
                        # prepare RRs
                        rrMsg = {'time': new_msg['time'], 'type': 1, 'from': new_msg['to'], 'to': new_msg['from'], 'UID': thisUID + 1}
                        read_receipts.append(rrMsg)

                # [NEW] add the messages in batches
                for gm in group_messages:
                    log.append(gm)
                # [NEW] add the RR in order
                for rr in read_receipts:
                    thisUID += 1
                    check_duplicate_uid(thisUID)
                    rr['UID'] = thisUID
                    rr_log.append(rr)
                    print(f"[DEBUG UID] Assigned UID {thisUID} to Read Receipt {rr}")
                
                print(f"[DEBUG UID] Returning UID: {thisUID} from processMessage")
                return thisUID
        
    # if no group match i.e. sender is not in a group or poisson = 0:
    check_duplicate_uid(thisUID)
    msg['to'] = receiver_id
    message_tuple = (msg['time'], msg['from'], msg['to'], msg['UID'])

    print(f"[DEBUG UID, 1:1] Assigning UID: {thisUID} to message from {msg['from']} to {msg['to']}")

    if message_tuple not in seen_messages:
        log.append(msg.copy())
        seen_messages.add(message_tuple)

        if msg['from'] in {0, 1, 2} and msg['to'] in {0, 1, 2}:
            print(f"[LOG MESSAGE] {msg['from']} -> {msg['to']} | UID: {msg['UID']} | Time: {msg['time']}")

    # issue read receipt on normal message
    thisUID = processReadReceipt(msg, rr_log, thisUID)
    print(f"[DEBUG UID] Returning UID: {thisUID} from processMessage")
    return thisUID

def createLogStandard(numEpochs, msgsPerEpoch, graph, group_data, poisson_array, rr=True):
    # Generate messages over numEpochs many epochs, ensure group communication behavior 
    # Args:
    # > numEpochs (int): number of epochs 
    # > msgsPerEpoch (int): messages per epoch
    # > graph (igraph.Graph): the network graph
    # > group_data (dict):  {group_id: [member1, member2, ...]} <-- created by graph_gen_master_realistic.py
    # > poisson_array (list): list of poisson values, defines group msg activity
    # > rr (bool): whether to include read receipts

    # Return: (list) sorted log of messages and receipts

    LogStandard = []
    rrLogStandard = []
    seen_messages = set()
    time = 0
    #thisUID = 0
    # just in case of multiprocessting ussues...
    #thisUID = (os.getpid() * 1000000) + random.randint(1, 1000000)
    thisUID = 0
    prevMessages = []
    activeAgents = np.arange(len(graph.vs))
    print("Number of Active Agents:", len(graph.vs))
    rng = np.random.default_rng()

    # Identify super users if the environment requires it
    if CONFIG["super_users"]:
        super_users = select_super_users(graph, group_data)
    else:
        super_users = set()
    # Get user 0's additional group (if applicable)
    extra_group_for_user_0 = CONFIG.get("extra_group_for_user_0", None)

    for epoch in range(numEpochs):
        poisson_value = poisson_array[epoch]
        #senders = [(i,None) for i in rng.choice(activeAgents, msgsPerEpoch - poisson_value)]
        #senders = list(rng.choice(activeAgents, msgsPerEpoch - poisson_value, replace = False))
        senders = [(i, None) for i in rng.choice(activeAgents, msgsPerEpoch - poisson_value, replace=False)]
        print(f"\n Epoch {epoch}: Poisson Value = {poisson_value}, Generating Messages")

        # carrying through: reply messages in the *next* epoch 
        reply_messages = int(len(prevMessages) * (1/4))
        
        valid_choice = [msg for msg in prevMessages if msg.get('to') not in group_data and msg.get('from') not in group_data]

        # used for tracking group messages
        group_messages = set()

        # handle replies here 
        if reply_messages > 0 and valid_choice:
            replies = rng.choice(valid_choice, min(reply_messages, len(valid_choice)), replace=False)
            for msg in replies: 
                senders.append((int(msg['to']), int(msg['from'])))
                #print(f"\n [REPLY] epoch {epoch}: Replying to message from {msg['from']} to {msg['to']}")

        # inject group conversation if Poisson value > 0 
        for _ in range(poisson_value):
            selected_group = random.choice(list(group_data.values()))
            sender, receiver = map(int, rng.choice(selected_group, size=2, replace=False))
            insert_index = rng.choice(range(msgsPerEpoch), replace=False)
            senders.insert(insert_index, (sender, receiver))
            group_messages.add((sender, receiver)) 

            # check if test group
            if set(selected_group) == {0, 1, 2}:
                print(f"\n [GROUP MSG] Epoch {epoch}: injected group message for (0,1,2)")
                print(f"Sender: {sender}, Receiver: {receiver}")
        
        print(f"Group Messages Added: {poisson_value}")

        new_messages = []

        for sender, receiver in senders:
            sender, receiver = int(sender), None if receiver is None else int(receiver)
            # handle the super users
            if sender in super_users:
                extra_messages = [(sender, int(rng.choice(graph.neighbors(sender)))) for _ in range(CONFIG["boost_factor"])]
                rng.shuffle(extra_messages)
                for extra_sender, extra_receiver in extra_messages:
                    # re-ordered next two lines
                    thisUID += 1
                    msg = {'time': int(time), 'type': 0, 'from': extra_sender, 'to': extra_receiver, 'UID': thisUID}
                    message_tuple = (time, msg['from'], msg['to'], msg['UID'])
                    
                    if message_tuple not in seen_messages:
                        LogStandard.append(msg.copy())
                        seen_messages.add(message_tuple)
                    # next 3 lines are new
                    #msg_copy = msg.copy()  # Ensure fresh copy
                    #msg['UID'] = thisUID  #moved and modified 
                    #thisUID += 1  #  Increment before using # NOT NEEDED
                    new_messages.append(msg.copy())
                    thisUID = processMessage(extra_sender, extra_receiver, msg, group_data, LogStandard, seen_messages, thisUID, poisson_value, rrLogStandard)

            # special case: user 0 super user in non-injected group
            if sender == 0 and extra_group_for_user_0:
                # generating multiple messages: 0 --> group
                extra_messages = [(0, random.choice([user for user in extra_group_for_user_0 if user != 0])) for _ in range(CONFIG["boost_factor"])]
                rng.shuffle(extra_messages)

                for extra_sender, extra_receiver in extra_messages:
                    thisUID += 1
                    msg = {'time': int(time), 'type': 0, 'from': extra_sender, 'to': extra_receiver, 'UID': thisUID}
                    message_tuple = (time, msg['from'], msg['to'], msg['UID'])

                    if message_tuple not in seen_messages:
                        LogStandard.append(msg.copy())
                        seen_messages.add(message_tuple)
                    # next 3 lines are new
                    #msg_copy = msg.copy()  # Ensure fresh copy
                    #thisUID += 1  #  Increment before using <<-- get rid of this?
                    #msg_copy['UID'] = thisUID  # Assign UID
                    new_messages.append(msg.copy())
                    thisUID = processMessage(extra_sender, extra_receiver, msg, group_data, LogStandard, seen_messages, thisUID, poisson_value, rrLogStandard)
                    
            is_group_message = any({sender, receiver}.issubset(set(members)) for members in group_data.values())
            # Recall the basic functioning: for a user implementing this defense, on *any* group message they send, 
            # they send DEF_STR many messages to themselves, producing corresponding receipts (msgs placed randomly in log)
            # TODO: unclear if this is actually going to work since the log must be sorted by UID ?
            if DEF == "self" and DEF_STR is not None and sender == 0 and is_group_message:
                print(f"[DEFENSE] Injecting {DEF_STR} self-messages for user 0")
                try:
                    def_strength = int(DEF_STR)
                    if def_strength < 1 or def_strength > 10:
                        print("[ERROR] Defense strength must be between 1 and 10. Skipping injection.")
                        continue
                except ValueError:
                    print("[ERROR] invalid defense strength argument. Skipping injection.")
                    continue
                self_messages =[]
                for _ in range(int(def_strength)):
                    # Generate self messages
                    self_msg = {'time': int(time), 'type': 0, 'from': 0, 'to': 0, 'UID': thisUID}
                    thisUID += 1
                    self_msg['UID'] = thisUID
                    # and their corresponding RR
                    rr_msg = {'time': int(time), 'type': 1, 'from': 0, 'to': 0, 'UID': thisUID}
                    thisUID += 1
                    rr_msg['UID'] = thisUID

                    self_messages.append((self_msg, rr_msg))

                for sm, rr_sm in self_messages:
                    # insert messages randomly throughout the log
                    insert_idx = rng.integers(0, len(new_messages) + 1)
                    new_messages.insert(insert_idx, sm)
                    LogStandard.append(sm.copy())
                    seen_messages.add((sm['time'], sm['from'], sm['to'], sm['UID']))
                    print(f"[DEBUG] Injected Self-Message at index {insert_idx}: {sm}")

                    new_messages.insert(insert_idx + 1, rr_sm)
                    rrLogStandard.append(rr_sm.copy())
                    seen_messages.add(((rr_sm['time'], rr_sm['from'], rr_sm['to'], rr_sm['UID'])))
                    print(f"[DEBUG] Injected Read Receipt it index {insert_idx + 1}: {rr_sm}")

            # normal message processing
            msg = {'time': int(time), 'type': 0, 'from': sender, 'UID': thisUID}
            thisUID += 1
            msg['UID'] = thisUID

            if receiver is None:
                try:
                    receiver = int(rng.choice(graph.neighbors(int(sender))))
                except ValueError:
                    continue

            msg['to'] = receiver
            message_tuple = (time, msg['from'], msg['to'], msg['UID'])

            #if not is_group_message and message_tuple not in seen_messages:
            if message_tuple not in seen_messages:
                LogStandard.append(msg.copy())
                seen_messages.add(message_tuple)
                
            new_messages.append(msg.copy())

            # read receipt
            #thisUID = processReadReceipt(msg, rrLogStandard, thisUID)

            # process message
            thisUID = processMessage(sender, receiver, msg, group_data, LogStandard, seen_messages, thisUID, poisson_value, rrLogStandard)

            if sender in {0,1,2} and receiver in {0,1,2}:
                print(f"\n [GROUP MSG] Epoch {epoch}: group message for (0,1,2)")
                print(f"Sender: {sender}, Receiver: {receiver}")
                print(f"Message: {msg}")

        prevMessages = new_messages[:]
        time += 1

        # debugging 
        if epoch % 10 == 0:
            print(f"\n [DEBUG] Epoch {epoch} - Sample Messages (100 entries):")
            for msg in LogStandard[-100:]:  # Slice the last 100 messages
                print(msg)
    
    fullLogStandard = sorted(LogStandard + rrLogStandard, key=lambda x: x['UID'])
    print(f"Senders: {senders}")
    print("Preliminary Success")
    return fullLogStandard


def processReadReceipt(msg, rr_log, thisUID):
    # generate read receipts (delivered receipt)
    # Args:
    # > msg (dict): the original message
    # >rr_log (list): Log where the receipts are stored
    # thisUID (int): UID counter
    # Return: (int), updated UID
    thisUID += 1
    check_duplicate_uid(thisUID)
    rrMsg = {'time': msg['time'], 'type': 1, 'from': msg['to'], 'to': msg['from'], 'UID': thisUID}
    #print(f"[DEBUG UID] Assigning UID: {thisUID} to message from {msg['from']} to {msg['to']}")
    rr_log.append(rrMsg.copy())
    print(f"[DEBUG UID] processReadReceipt returning UID: {thisUID}")
    return thisUID


def task(args):
    # moving the code for generation and attack from the main function to ``task" in order to multiprocess the simulation
    group_data, attack_type = args

    rng = np.random.default_rng()
    graph_path = os.path.expanduser(f"~/signalsim/{GRAPH_FILE}.tmp")
    g = Graph.Read_Edgelist(graph_path)
    msgsPerEpoch = 800
    # use numEpochs to dictate the length of the test 9k ==> 20 flurries per 3 person group, 6k ==> 15 flurries per 3 person group (reliably)
    #numEpochs = 1000
    numEpochs = 9000

    # poisson arrays
    #lamda_param1 = .01
    #lamda_param2 = .01
    lambda_param1 = 2    
    #lambda_param2 = .1
    poisson_array1 = generatePoissonValues(lambda_param1, numEpochs+10)
    #poisson_array2 = generatePoissonValues(lambda_param2, numEpochs+10)

    LogStandard = createLogStandard(numEpochs, msgsPerEpoch, g, group_data, poisson_array1)

    # choose the attack type based on user input to --atk
    # TODO: 1) modify the preamble to include attack scripts 
    #       2) modify the code after this block to be attack_module.(function)
    ATTACK_MODULES = {
        "standard": notime_cor_attack_variable_flurries_standard,
        "large": notime_cor_attack_variable_flurries_large,
        "unrel_super": notime_cor_attack_variable_flurries_popular,
        "group_super": notime_cor_attack_variable_flurries_multi_popular_group_user,
        "multi_disjoint": notime_cor_attack_variable_flurries_with,
        "multi_over_equal": notime_cor_attack_variable_flurries_with,
        "mult_over_super": notime_cor_attack_variable_flurries_with_popular,
        "multimulti": notime_cor_attack_variable_flurries_multi,
        "martiny": martiny_attack_recreate,
    }
    # select attack module 
    attack_module = ATTACK_MODULES.get(attack_type)
    if attack_module is None:
        print(f"[ALERT] Attack {attack_type} does not exist. Try again.")
        return None

    # group compositions
    standard_group = {0, 1, 2}
    large_group = {0, 1, 2, 3, 4}
    
    # standard group
    numFlurriesInspect0 = attack_module.detectFlurry(LogStandard, 0)
    numFlurriesInspect1 = attack_module.detectFlurry(LogStandard, 1)
    numFlurriesInspect2 = attack_module.detectFlurry(LogStandard, 2)

    ranks0, ranks1, ranks2 = [], [], []

    # set the number of flurries to inspect here
    for _ in range (1, 21, 1):
        result = attack_module.attack(LogStandard, 0, _)
        result1 = attack_module.attack(LogStandard, 1, _)
        result2 = attack_module.attack(LogStandard, 2, _)

        ranks0.append(attack_module.getMaxRank(result, standard_group - {0}))
        ranks1.append(attack_module.getMaxRank(result1, standard_group - {1}))
        ranks2.append(attack_module.getMaxRank(result2, standard_group - {2}))

    if EXPERIMENT == "large":
        numFlurriesInspect3 = attack_module.detectFlurry(LogStandard, 3)
        numFlurriesInspect4 = attack_module.detectFlurry(LogStandard, 4)

        ranks3, ranks4 = [], []

        for _ in range (1, 21, 1):
            result3 = attack_module.getMaxRank(LogStandard, 3, _)
            result4 = attack_module.getMaxRank(LogStandard, 4, _)

            ranks3.append(attack_module.getMaxRank(result3, large_group - {3}))
            ranks3.append(attack_module.getMaxRank(result3, large_group - {4}))

        return (ranks0, ranks1, ranks2, numFlurriesInspect0, numFlurriesInspect1, numFlurriesInspect2,
                ranks3, ranks4, numFlurriesInspect3, numFlurriesInspect4)

    return (ranks0, ranks1, ranks2, numFlurriesInspect0, numFlurriesInspect1, numFlurriesInspect2)

def main():
    flurriesToObserve = 21
    x_axis = list(range(1,flurriesToObserve,1))
    marker_size = 20
    #runs = 20
    runs = 1

    group_file = os.path.expanduser("~/signalsim/group_member_counts.tsv")
    group_data = genGroups(group_file) #test_group_id just for unpacking, not using 

    # current experiments
    print(f"Messaging style: {CURRENT_ENV}")
    print(f"Working on graph: {GRAPH_FILE}")
    print(f"Performing Experiment: {EXPERIMENT}")
    print(f"Running AttackL {ATTACK_TYPE}")
    print(f"With Defense: {DEF} and strength {DEF_STR}")

    # worker processes 
    with multiprocessing.Pool() as pool:
        #results = pool.map(task, [(i, group_data[0]) for  i in range(runs)])
        results = pool.map(task, [(group_data[0], ATTACK_TYPE) for i in range(runs)])

    y_axis_runs0, y_axis_runs1, y_axis_runs2 = [], [], []
    numFlurryArr0, numFlurryArr1, numFlurryArr2 = [], [], []

    y_axis_runs3, y_axis_runs4 = [], []
    numFlurryArr3, numFlurryArr4 = [], []

    for result in results:
        if EXPERIMENT == "large":
            (ranks0, ranks1, ranks2, numFlurriesInspect0, numFlurriesInspect1, numFlurriesInspect2, 
            ranks3, ranks4, numFlurriesInspect3, numFlurriesInspect4) = result
            y_axis_runs3.append(ranks3)
            y_axis_runs4.append(ranks4)
            numFlurryArr3.append(numFlurriesInspect3)
            numFlurryArr4.append(numFlurriesInspect4)
        else:
            ranks0, ranks1, ranks2, numFlurriesInspect0, numFlurriesInspect1, numFlurriesInspect2 = result
        
        y_axis_runs0.append(ranks0)
        y_axis_runs1.append(ranks1)
        y_axis_runs2.append(ranks2)

        numFlurryArr0.append(numFlurriesInspect0)
        numFlurryArr1.append(numFlurriesInspect1)
        numFlurryArr2.append(numFlurriesInspect2)

    # compute averages
    averages = [sum(col) / len(col) for col in zip(*y_axis_runs0)]
    averages1 = [sum(col) / len(col) for col in zip(*y_axis_runs1)]
    averages2 = [sum(col) / len(col) for col in zip(*y_axis_runs2)]

    print("ENTRIES IN RANKS: " + str(len(ranks0)))
    print("ENTRIES IN X-AXIS: " + str(len(x_axis)))
    print("Flurry detection counts of user 0: " + str(numFlurryArr0))
    print("Flurry detection counts of user 1: " + str(numFlurryArr1))
    print("Flurry detection counts of user 2: " + str(numFlurryArr2))

    # Save results to a text file
    results_filename = f"results_{EXPERIMENT}.txt"
    with open(results_filename, "w") as f:
        f.write(f"Experiment: {EXPERIMENT}\n")
        f.write(f"Flurries To Observe: {flurriesToObserve - 1}\n")
        f.write(f"Runs: {runs}\n")
        f.write("\nRanks:\n")
        f.write("Target 0: " + ", ".join(map(str, averages)) + "\n")
        f.write("Target 1: " + ", ".join(map(str, averages1)) + "\n")
        f.write("Target 2: " + ", ".join(map(str, averages2)) + "\n")
        f.write("\nFlurry Detection Counts:\n")
        f.write("User 0: " + ", ".join(map(str, numFlurryArr0)) + "\n")
        f.write("User 1: " + ", ".join(map(str, numFlurryArr1)) + "\n")
        f.write("User 2: " + ", ".join(map(str, numFlurryArr2)) + "\n")

        if EXPERIMENT == "large":
            averages3 = [sum(col) / len(col) for col in zip(*y_axis_runs3)]
            averages4 = [sum(col) / len(col) for col in zip(*y_axis_runs4)]
            f.write("Target 3: " + ", ".join(map(str, averages3)) + "\n")
            f.write("Target 4: " + ", ".join(map(str, averages4)) + "\n")
            f.write("User 3: " + ", ".join(map(str, numFlurryArr3)) + "\n")
            f.write("User 4: " + ", ".join(map(str, numFlurryArr4)) + "\n")

    print(f"Results saved to {results_filename}")
    
    plt.scatter(x_axis, averages, s=marker_size, label = 'Target 0', marker='+', color='black')  
    plt.scatter(x_axis, averages1, s=marker_size, label = 'Target 1', marker='o', edgecolor='black', facecolor='none')  
    plt.scatter(x_axis, averages2, s=marker_size, label = 'Target 2', marker='s', edgecolor='black', facecolor='none')  
    
    if EXPERIMENT == "large":
        averages3 = [sum(col) / len(col) for col in zip(*y_axis_runs3)]
        averages4 = [sum(col) / len(col) for col in zip(*y_axis_runs4)]
        plt.scatter(x_axis, averages3, s=marker_size, label = 'Target 3', marker='D', edgecolor='black', facecolor='none')
        plt.scatter(x_axis, averages4, s=marker_size, label = 'Target 4', marker='^', edgecolor='black', facecolor='none')
        
    
    
    #plt.scatter(x_axis, averagesUnrel, s=marker_size, label = 'Non-Group User', marker='1', color='red')
    #plt.axis('scaled')
    # LogStandard scale... 
    plt.yscale('log', base=10) 
    plt.title(f'Average Flurries Needed to Uncover Groups -- 100K Users, {runs} Trials')
    plt.xlabel('Number of Flurries Observed')
    plt.ylabel('Average Max Rank of Known Group Members')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    for _ in range(20):
        print('\007')
    plt.savefig(f"plot_{EXPERIMENT}.png")
    plt.show()

if __name__ == "__main__":
    main()