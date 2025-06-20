from ipaddress import collapse_addresses
from operator import truediv
from pickle import TRUE
from plistlib import UID
from igraph import *
import json
import attack as atk
import cor_attack as cor_atk
# attack imports TODO: improve this -- stretch goal
import updated_attack
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
from graph_gen_master_realistic import genGroups, genMartinyGroups
import os
import argparse

def LogStandardSort(d):
    return d['time']

def UIDSort(p):
    return p['UID']

ENVIRONMENTS = {
    "default": { "super_users": False},
    # the next environment is for super users among the entire network, not necessarily belonging to the group
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
parser.add_argument("--mar", type=lambda x: (str(x).lower() == 'true'), required=False, default=False, help="Enable Martiny-style gen+attack (True/False).")
args = parser.parse_args()

CURRENT_ENV = args.env
GRAPH_FILE = args.graph
EXPERIMENT = args.exp
ATTACK_TYPE = args.atk
DEF = args.defs
DEF_STR = args.str
MARTINY = args.mar
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
    elif CONFIG.get("group_super_user", False):
        super_users.add(random.choice([0, 1, 2]))
    # on super_user_ratio*TotalUsers many super users
    elif CONFIG["super_users"]:
        eligible_users = set(range(len(graph.vs))) - {0, 1, 2, 3, 4, 5}
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
                        #print(f"[DEBUG] Checking UID {msg['UID']} before forwarding {sender_id} -> {receiver_id}")
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

                        #print(f"[DEBUG UID, GROUP] Assigning UID: {thisUID} to message from {new_msg['from']} to {new_msg['to']}")

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
                    #print(f"[DEBUG UID] Assigned UID {thisUID} to Read Receipt {rr}")
                
                #print(f"[DEBUG UID] Returning UID: {thisUID} from processMessage")
                return thisUID
        
    # if no group match i.e. sender is not in a group or poisson = 0:
    check_duplicate_uid(thisUID)
    msg['to'] = receiver_id
    message_tuple = (msg['time'], msg['from'], msg['to'], msg['UID'])

    #print(f"[DEBUG UID, 1:1] Assigning UID: {thisUID} to message from {msg['from']} to {msg['to']}")

    if message_tuple not in seen_messages:
        log.append(msg.copy())
        seen_messages.add(message_tuple)

        if msg['from'] in {0, 1, 2} and msg['to'] in {0, 1, 2}:
            print(f"[LOG MESSAGE] {msg['from']} -> {msg['to']} | UID: {msg['UID']} | Time: {msg['time']}")

    # issue read receipt on normal message
    thisUID = processReadReceipt(msg, rr_log, thisUID)
    #print(f"[DEBUG UID] Returning UID: {thisUID} from processMessage")
    return thisUID

def injectSelfMessage(epoch, time, thisUID, self_defense_array, group_data, LogStandard, rrLogStandard, seen_messages, seen_uids):
    # injecting self messages for user 0, routing through processMessage (which takes care of receipt)
    # returns updated UID
    num_self_msgs = self_defense_array[epoch]
    if num_self_msgs > 0:
        print(f"[SELF-DEFENSE] Epoch {epoch}: injecting {num_self_msgs} self messages")
        for _ in range(num_self_msgs):
            thisUID += 1
            msg = {'time': int(time), 'type': 0, 'from': 0, 'to': 0, 'UID': thisUID}
            thisUID = processMessage(0, 0, msg, group_data, LogStandard, seen_messages, thisUID, 0, rrLogStandard)
    return thisUID

def createLogStandard(numEpochs, msgsPerEpoch, graph, group_data, poisson_array, self_defense_array, rr=True, super_user_poisson_map=None):
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
    thisUID = 0
    prevMessages = []
    activeAgents = np.arange(len(graph.vs))
    print("Number of Active Agents:", len(graph.vs))
    rng = np.random.default_rng()

    # Identify super users if the environment requires it
    #if CONFIG["super_users"]:
    #    super_users = select_super_users(graph, group_data)
    #else:
    #    super_users = set()
    # Get user 0's additional group (if applicable)
    extra_group_for_user_0 = CONFIG.get("extra_group_for_user_0", None)

    for epoch in range(numEpochs):
        poisson_value = poisson_array[epoch]
        #senders = [(i,None) for i in rng.choice(activeAgents, msgsPerEpoch - poisson_value)]
        #senders = list(rng.choice(activeAgents, msgsPerEpoch - poisson_value, replace = False))
        senders = [(i, None) for i in rng.choice(activeAgents, msgsPerEpoch - poisson_value, replace=False)]
        print(f"\n Epoch {epoch}: Poisson Value = {poisson_value}, Generating Messages")
        uid_counter = 0
        defense_triggered_this_epoch = False

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
            if set(selected_group) == {0, 1, 2, 3, 4}:
                print(f"\n [GROUP MSG] Epoch {epoch}: injected group message for (0,1,2,3,4)")
                print(f"Sender: {sender}, Receiver: {receiver}")
        
        print(f"Group Messages Added: {poisson_value}")

        new_messages = []

        for sender, receiver in senders:
            sender, receiver = int(sender), None if receiver is None else int(receiver)
            # handle the super users
            #if sender in super_users:
            #    extra_messages = [(sender, int(rng.choice(graph.neighbors(sender)))) for _ in range(CONFIG["boost_factor"])]
            #    rng.shuffle(extra_messages)
            #    for extra_sender, extra_receiver in extra_messages:
            #        # re-ordered next two lines
            #        thisUID += 1
            #        msg = {'time': int(time), 'type': 0, 'from': extra_sender, 'to': extra_receiver, 'UID': thisUID}
            #        message_tuple = (time, msg['from'], msg['to'], msg['UID'])
            #        
            #        if message_tuple not in seen_messages:
            #            LogStandard.append(msg.copy())
            #            seen_messages.add(message_tuple)
            #        # next 3 lines are new
            #        #msg_copy = msg.copy()  # Ensure fresh copy
            #        #msg['UID'] = thisUID  #moved and modified 
            #        #thisUID += 1  #  Increment before using # NOT NEEDED
            #        new_messages.append(msg.copy())
            #        thisUID = processMessage(extra_sender, extra_receiver, msg, group_data, LogStandard, seen_messages, thisUID, poisson_value, rrLogStandard)

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
            if sender == 0:
                print(f"[ZERO SENDER]: {sender} --> {receiver}")

        # Inject self-messages for user 0 based on Poisson process
        if DEF == "self" and DEF_STR is not None:
            thisUID = injectSelfMessage(epoch, time, thisUID, self_defense_array, group_data, LogStandard, rrLogStandard, seen_messages, seen_uids)

        # Inject messages for each super user according to their Poisson process
        if super_user_poisson_map:
            print("inside super user condition")
            for super_user, poisson_values in super_user_poisson_map.items():
                count = poisson_values[epoch]
                for _ in range(count):
                    try:
                        recipient = int(rng.choice(graph.neighbors(super_user)))
                    except ValueError:
                        continue  # No neighbors, skip

                    msg = {
                        "time": int(time),
                        "type": 0,
                        "from": super_user,
                        "to": recipient,
                        "UID": thisUID
                    }
                    thisUID += 1
                    msg['UID'] = thisUID
                    tup = (time, msg['from'], msg['to'], msg['UID'])

                    if tup not in seen_messages:
                        LogStandard.append(msg.copy())
                        seen_messages.add(tup)

                    new_messages.append(msg.copy())
                    #print(f"Added super user messages: {msg}")
                    thisUID = processMessage(super_user, recipient, msg, group_data, LogStandard, seen_messages, thisUID, poisson_value, rrLogStandard)
        else:
            print("Super User Injection Failed.")

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
    #print(f"[DEBUG UID] processReadReceipt returning UID: {thisUID}")
    return thisUID

def find_top_senders_to_zero(log, martiny_contacts):
    """
    Find the martiny_contacts who sent messages to user 0,
    and drop the user who sent the fewest messages.

    Args:
        log (list): List of message dictionaries.
        martiny_contacts (set): Set of user IDs to check.

    Returns:
        set: Updated martiny_contacts with the least active sender removed.
    """
    from collections import Counter

    counter = Counter()

    for msg in log:
        if msg['type'] == 0 and msg['to'] == 0 and msg['from'] in martiny_contacts:
            counter[msg['from']] += 1

    # Ensure every martiny_contact appears (even if they sent 0 messages)
    for contact in martiny_contacts:
        counter.setdefault(contact, 0)

    if not counter:
        print("[WARNING] No messages from martiny contacts to user 0 found.")
        return martiny_contacts.copy()

    # NEW: Print counts for all martiny_contacts
    print("[INFO] Message counts for martiny contacts to user 0:")
    for user in sorted(martiny_contacts):
        print(f"User {user}: {counter[user]} messages")

    # Find minimum count
    min_count = min(counter.values())
    users_with_min = [user for user, count in counter.items() if count == min_count]

    # Pick one user to drop (could randomize; here just pick the first)
    user_to_drop = users_with_min[0]

    updated_contacts = martiny_contacts.copy()
    updated_contacts.discard(user_to_drop)

    print(f"[INFO] Dropping user {user_to_drop} who sent {min_count} messages to user 0.")
    return updated_contacts


def task(args):
    # moving the code for generation and attack from the main function to ``task" in order to multiprocess the simulation
    group_data, attack_type, unrelated_user = args

    rng = np.random.default_rng()
    graph_path = os.path.expanduser(f"~/signalsim/{GRAPH_FILE}.tmp")
    g = Graph.Read_Edgelist(graph_path)
    msgsPerEpoch = 800
    # use numEpochs to dictate the length of the test 9k ==> 20 flurries per 3 person group, 6k ==> 15 flurries per 3 person group (reliably)
    #numEpochs = 1000
    #numEpochs = 2000
    #numEpochs = 5000
    numEpochs = 18000
    #numEpochs = 12000

    # poisson arrays
    #lamda_param1 = .01
    #lamda_param2 = .01
    # group lambda value
    lambda_param1 = 1.4    
    #lambda_param2 = .1
    if DEF is not None:
        self_def_param = int(DEF_STR)*0.008
        self_defense_array = generatePoissonValues(self_def_param, numEpochs+10)
    else:
        self_defense_array = [] * (numEpochs+10)
    poisson_array1 = generatePoissonValues(lambda_param1, numEpochs+10)
    #poisson_array2 = generatePoissonValues(lambda_param2, numEpochs+10)

    super_user_poisson_map = {}
    super_user_agents = {}

    if CONFIG.get("super_users", False):
        super_user_agents = select_super_users(g, group_data)
        super_lambda = 0.008 * CONFIG["boost_factor"]
        for user in super_user_agents:
            super_user_poisson_map[user] = generatePoissonValues(super_lambda, numEpochs + 10)
    else:
        super_user_poisson_map = None  # <- so you don't accidentally inject extra messages

    LogStandard = createLogStandard(numEpochs, msgsPerEpoch, g, group_data, poisson_array1, self_defense_array, super_user_poisson_map=super_user_poisson_map)

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
        "updated": updated_attack,
    }
    # select attack module 
    attack_module = ATTACK_MODULES.get(attack_type)
    if attack_module is None:
        print(f"[ALERT] Attack {attack_type} does not exist. Try again.")
        return None
    
    if MARTINY:
        _ = attack_module.show_messages_to_from_zero(LogStandard)
        martiny_contacts = {3, 4, 5, 6, 7}
        updated_martiny_contacts = find_top_senders_to_zero(LogStandard, martiny_contacts)
        print(f"New martiny contacts set: {updated_martiny_contacts}")
        martiny_max_ranks = []
        martiny_min_ranks = []
        for epoch in range(1, 201, 1):
            martiny_result = attack_module.martiny_attack(LogStandard, 0, epoch)
            if martiny_result is None:
                max_rank = -1
                min_rank = -1
            else:
                max_rank = attack_module.getMaxRank(martiny_result, updated_martiny_contacts)
                min_rank = attack_module.getMinRank(martiny_result, updated_martiny_contacts)
            martiny_max_ranks.append(max_rank)
            martiny_min_ranks.append(min_rank)
        return[martiny_max_ranks, martiny_min_ranks]

    # group compositions
    standard_group = {0, 1, 2}
    large_group = {0, 1, 2, 3, 4}
    
    # standard group
    numFlurriesInspect0 = attack_module.detectFlurry(LogStandard, 0)
    numFlurriesInspect1 = attack_module.detectFlurry(LogStandard, 1)
    numFlurriesInspect2 = attack_module.detectFlurry(LogStandard, 2)

    ranks_unrel = []
    ranks0, ranks1, ranks2 = [], [], []
    results0, results1, results2 = [], [], []

    numFlurriesInspect_unrel = attack_module.detectFlurry(LogStandard, unrelated_user)

    if DEF is not None and unrelated_user is not None:
        for _ in range(1, 21):
            result_unrel = attack_module.attack(LogStandard, unrelated_user, _)
            ranks_unrel.append(attack_module.getMaxRank(result_unrel, {0,1,2}))

    # set the number of flurries to inspect here
    for _ in range (1, 21):
        result = attack_module.attack(LogStandard, 0, _)
        result1 = attack_module.attack(LogStandard, 1, _)
        result2 = attack_module.attack(LogStandard, 2, _)

        results0.append(result)
        results1.append(result1)
        results2.append(result2)

        ranks0.append(attack_module.getMaxRank(result, standard_group - {0}))
        ranks1.append(attack_module.getMaxRank(result1, standard_group - {1}))
        ranks2.append(attack_module.getMaxRank(result2, standard_group - {2}))

        print(f"Flurry {_}: MaxRank0={ranks0[-1]}, MaxRank1={ranks1[-1]}, MaxRank2={ranks2[-1]}")

    if EXPERIMENT == "large":
        numFlurriesInspect3 = attack_module.detectFlurry(LogStandard, 3)
        numFlurriesInspect4 = attack_module.detectFlurry(LogStandard, 4)

        ranks0, ranks1, ranks2 = [], [], []
        ranks3, ranks4 = [], []

        for _ in range (1, 21, 1):
            ranks0.append(attack_module.getMaxRank(results0[_ - 1], large_group - {0}))
            ranks1.append(attack_module.getMaxRank(results1[_ - 1], large_group - {1}))
            ranks2.append(attack_module.getMaxRank(results2[_ - 1], large_group - {2}))

            result3 = attack_module.attack(LogStandard, 3, _)
            result4 = attack_module.attack(LogStandard, 4, _)

            ranks3.append(attack_module.getMaxRank(result3, large_group - {3}))
            ranks4.append(attack_module.getMaxRank(result4, large_group - {4}))

        return (ranks0, ranks1, ranks2, numFlurriesInspect0, numFlurriesInspect1, numFlurriesInspect2,
                ranks3, ranks4, numFlurriesInspect3, numFlurriesInspect4)

    return (ranks0, ranks1, ranks2, numFlurriesInspect0, numFlurriesInspect1, numFlurriesInspect2, ranks_unrel, numFlurriesInspect_unrel)

def main():
    flurriesToObserve = 21
    x_axis = list(range(1,flurriesToObserve,1))
    marker_size = 20
    #runs = 1
    #runs = 5
    runs = 10
    #runs = 20

    group_file = os.path.expanduser("~/signalsim/group_member_counts.tsv")
    group_data, _ = genGroups(group_file) #test_group_id just for unpacking, not using 
    martinyGroup_data = genMartinyGroups(group_file)

    if DEF is not None:
        all_group_members = set()
        for members in group_data.values():
            all_group_members.update(members)
        candidate_users = [user for user in all_group_members if user not in {0, 1, 2, 3, 4}]

        if not candidate_users:
            raise ValueError("No suitable unrelated users found in any group.")
        
        unrelated_user =  random.choice(candidate_users)
        print(f"Random unrelated user selected: {unrelated_user}")
    else:
        unrelated_user = None

    # current experiments
    print(f"Messaging style: {CURRENT_ENV}")
    print(f"Working on graph: {GRAPH_FILE}")
    print(f"Performing Experiment: {EXPERIMENT}")
    print(f"Running Attack: {ATTACK_TYPE}")
    print(f"With Defense: {DEF} and strength {DEF_STR}")
    print(f"Martiny: {MARTINY}")

    # worker processes 
    with multiprocessing.Pool() as pool:
        #results = pool.map(task, [(i, group_data[0]) for  i in range(runs)])
        if MARTINY:
            results = pool.map(task, [(martinyGroup_data, ATTACK_TYPE, unrelated_user) for i in range(runs)])
        else:
            results = pool.map(task, [(group_data, ATTACK_TYPE, unrelated_user) for i in range(runs)])

    y_axis_runs_unrel = [] if DEF is not None else None
    y_axis_runs0, y_axis_runs1, y_axis_runs2 = [], [], []
    numFlurryArr0, numFlurryArr1, numFlurryArr2 = [], [], []

    y_axis_runs3, y_axis_runs4 = [], []
    numFlurryArr3, numFlurryArr4 = [], []
    numFlurryArr_unrel = []
    ranks_after_20_0 = []

    martiny_all_runs_max_ranks = []
    martiny_all_runs_min_ranks = []

    for result in results:
        if MARTINY:
            martiny_all_runs_max_ranks.append(result[0])  # Append max_ranks from this result
            martiny_all_runs_min_ranks.append(result[1])  # Append min_ranks from this result
            continue  # Important: don't run the rest of the code for MARTINY case

        if EXPERIMENT == "large":
            (ranks0, ranks1, ranks2, numFlurriesInspect0, numFlurriesInspect1, numFlurriesInspect2, 
            ranks3, ranks4, numFlurriesInspect3, numFlurriesInspect4) = result
            y_axis_runs3.append(ranks3)
            y_axis_runs4.append(ranks4)
            numFlurryArr3.append(numFlurriesInspect3)
            numFlurryArr4.append(numFlurriesInspect4)
        else:
            ranks0, ranks1, ranks2, numFlurriesInspect0, numFlurriesInspect1, numFlurriesInspect2, ranks_unrel, numFlurriesInspect_unrel = result
        
        if DEF is not None:
            y_axis_runs_unrel.append(ranks_unrel)
            numFlurryArr_unrel.append(numFlurriesInspect_unrel)
        
        y_axis_runs0.append(ranks0)
        y_axis_runs1.append(ranks1)
        y_axis_runs2.append(ranks2)

        numFlurryArr0.append(numFlurriesInspect0)
        numFlurryArr1.append(numFlurriesInspect1)
        numFlurryArr2.append(numFlurriesInspect2)

        if len(ranks0) >= 20:
            ranks_after_20_0.append(ranks0[19])

    if MARTINY:
        martiny_max_averages = [sum(col) / len(col) for col in zip(*martiny_all_runs_max_ranks)]
        martiny_min_averages = [sum(col) / len(col) for col in zip(*martiny_all_runs_min_ranks)]

        # Save results to a file
        results_filename = f"results_martiny_{ATTACK_TYPE}.txt"
        with open(results_filename, "a") as f:
            f.write(f"Experiment: {EXPERIMENT}\n")
            f.write(f"Attack Type: {ATTACK_TYPE}\n")
            f.write(f"Environment: {CURRENT_ENV}\n")
            f.write(f"Martiny Mode: True\n")    
            f.write(f"Defense: {DEF}, Strength: {DEF_STR}\n")
            f.write(f"Runs: {runs}, Obs: 200 \n")
            f.write(f"With 5 Contacts, Restricting to 4\n")
            f.write("Martiny Max Rank Averages:\n")
            f.write(", ".join(map(str, martiny_max_averages)) + "\n")
            f.write("Martiny Min Rank Averages:\n")
            f.write(", ".join(map(str, martiny_min_averages)) + "\n")

        print(f"Martiny results saved to {results_filename}")
        return

    # compute averages
    averages = [sum(col) / len(col) for col in zip(*y_axis_runs0)]
    averages1 = [sum(col) / len(col) for col in zip(*y_axis_runs1)]
    averages2 = [sum(col) / len(col) for col in zip(*y_axis_runs2)]
    if EXPERIMENT == "large":
        averages3 = [sum(col) / len(col) for col in zip(*y_axis_runs3)]
        averages4 = [sum(col) / len(col) for col in zip(*y_axis_runs4)]
    if DEF is not None:
        averages_unrel = [sum(col) / len(col) for col in zip(*y_axis_runs_unrel)]

    print("ENTRIES IN RANKS: " + str(len(ranks0)))
    print("ENTRIES IN X-AXIS: " + str(len(x_axis)))
    print("Flurry detection counts of user 0: " + str(numFlurryArr0))
    print("Flurry detection counts of user 1: " + str(numFlurryArr1))
    print("Flurry detection counts of user 2: " + str(numFlurryArr2))
    if EXPERIMENT == "large":
        print("Flurry detection counts of user 3: " + str(numFlurryArr3))
        print("Flurry detection counts of user 4: " + str(numFlurryArr4))
    if DEF is not None:
        print("Flurry detection counts of unrelated user: " + str(numFlurryArr_unrel))

    # Save results to a text file
    if DEF is not None:
        results_filename = f"results_{EXPERIMENT}_{DEF}_{DEF_STR}.txt"
    else:
        results_filename = f"results_{EXPERIMENT}_{CONFIG}.txt"
    with open(results_filename, "a") as f:
        f.write(f"Experiment: {EXPERIMENT}\n")
        f.write(f"Attack Type: {ATTACK_TYPE}\n")
        f.write(f"Environment Type: {CONFIG}\n")
        if DEF is not None:
            f.write(f"Defense Type: {DEF}, Strength: {DEF_STR}\n")
            f.write(f"Unrelated User Averages: {averages_unrel}\n")
            f.write(f"Max-Rank of Others (Def User 0): {ranks_after_20_0}\n")
        f.write(f"Flurries To Observe: {flurriesToObserve - 1}\n")
        f.write(f"Runs: {runs}\n")
        f.write("\nRanks:\n")
        f.write("Target 0: " + ", ".join(map(str, averages)) + "\n")
        f.write("Target 1: " + ", ".join(map(str, averages1)) + "\n")
        f.write("Target 2: " + ", ".join(map(str, averages2)) + "\n")
        if EXPERIMENT == "large":
            f.write("Target 3: " + ", ".join(map(str, averages3)) + "\n")
            f.write("Target 4: " + ", ".join(map(str, averages4)) + "\n")
        f.write("\nFlurry Detection Counts:\n")
        f.write("User 0: " + ", ".join(map(str, numFlurryArr0)) + "\n")
        f.write("User 1: " + ", ".join(map(str, numFlurryArr1)) + "\n")
        f.write("User 2: " + ", ".join(map(str, numFlurryArr2)) + "\n")
        if EXPERIMENT == "large":
            f.write("User 3: " + ", ".join(map(str, numFlurryArr3)) + "\n")
            f.write("User 4: " + ", ".join(map(str, numFlurryArr4)) + "\n")

    print(f"Results saved to {results_filename}")
    
    if DEF is None:
        plt.scatter(x_axis, averages, s=marker_size, label = 'Target 0', marker='+', color='black')  
        plt.scatter(x_axis, averages1, s=marker_size, label = 'Target 1', marker='o', edgecolor='black', facecolor='none')  
        plt.scatter(x_axis, averages2, s=marker_size, label = 'Target 2', marker='s', edgecolor='black', facecolor='none')  

    if EXPERIMENT == "large":
        plt.scatter(x_axis, averages3, s=marker_size, label = 'Target 3', marker='D', edgecolor='black', facecolor='none')
        plt.scatter(x_axis, averages4, s=marker_size, label = 'Target 4', marker='^', edgecolor='black', facecolor='none')
        
    if DEF is not None: 
        plt.scatter(x_axis, averages, s=marker_size, label = 'Target 0', marker='+', color='black')  
        plt.scatter(x_axis, averages1, s=marker_size, label = 'Target 1', marker='o', edgecolor='black', facecolor='none')  
        plt.scatter(x_axis, averages2, s=marker_size, label = 'Target 2', marker='s', edgecolor='black', facecolor='none')
        plt.scatter(x_axis, averages_unrel, s=marker_size, label = 'Non-Group User', marker='1', color='red')
    
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
    if DEF is not None:
        plt.savefig(f"plot_{EXPERIMENT}_{DEF}_{DEF_STR}.png")
    else:
        plt.savefig(f"plot_{EXPERIMENT}_{CONFIG}.png")
    plt.show()

if __name__ == "__main__":
    main()