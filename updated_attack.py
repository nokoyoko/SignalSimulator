import json
import numpy as np
import random
from collections import Counter

# README
# This attack is a modification of "attack.py" to more accurately reflect the funcitoning of the proposed 
# attack on paper. While attack.py executed a simple intersection, this will attempt to uncover groups
# probabilistically rather than perfectly by sampling target and random epochs. 
# ------      ------      ------      ------      ------      ------      ------      ------      ------
# Additionally, want the attack to no longer quit after no detection of a flurry...
# but we also probably do not need to to check everything, maybe just increment some 
# counter which we will quit after $n$ many flurries have not been seen 
# TODO: what should $n$ be? Does this effect our analysis?

# SEEMS FINE -- grabs messages sent to target
def findRealSender(messages, personOfInterest):
    for msg in messages:
        if msg['to'] == personOfInterest:
            return msg['from']




# seems fine, takes in a certain 'time' and finds the 
# time of the next message sent by the user, such that 
# time > 'time' -- returns 'None' if DNE 
def findTimeOfFirstMessageAfterTime(messages, personOfInterest, time):
    for msg in messages:
        if msg['from'] == personOfInterest and msg['time'] > time:
            print("SUCCESS: FOUND TIME")
            return msg['time']
    return None


#SEEMS FINE, list of ALL DRs
def readReceiptsAtTime(messages, time):
    rrs = []
    for msg in messages:
        if msg['time'] == time and msg['type'] == 1:
            rrs.append(msg)
    
    return rrs




#SEEMS FINE, list of normal messages (not DR)
def normalMessagesAtTime(messages, time):
    normalMsgs = []
    for msg in messages:
        if msg['time'] == time and msg['type'] == 0:
            normalMsgs.append(msg)
            #print("APPENDED NORMAL MESSAGE.")
    
    return normalMsgs



# returns the number of flurries present for a set of messages 'messages' and a target 'personOfInterest'
def detectFlurry(messages, personOfInterest):
    possFlurry =[]
    #delta = 50
    delta = 25
    #delta = 10
    targetMsgCount = 0
    #threshold  = group size - 1
    threshold = 2
    #threshold = 4
    numFlurries = 0
    #iterate over messages
    for msg in messages: 
        possFlurry.append(msg) 

    print("possFlurry size: " + str(len(possFlurry)))

    j = 0
    lastFlurryIndex = - delta
    while j < len(possFlurry):
        msg = possFlurry[j]
        if msg['to'] == personOfInterest:
            for k in range(j, min(j + delta + 1, len(possFlurry))):
                if possFlurry[k]['to'] == personOfInterest:
                    targetMsgCount += 1
                    # print("Increment targetMsgCount, currently: " + str(targetMsgCount))
                    if targetMsgCount == threshold:
                        # next lines added in an attempt to ensure we do not double count flurries 
                        if j - lastFlurryIndex >= delta:
                            numFlurries += 1
                            lastFlurryIndex = j
                            #print(f"FLURRY IDENTIFIED, INC numFlurries to {numFlurries}")
                        j += delta
                        break
        j += 1
        targetMsgCount = 0
    print(f"numFlurries: {numFlurries}")   
    return numFlurries



## is the look ahead window built correctly?

# graphs for differing epoch lengths 
def buildTargetEpoch(messages, personOfInterest, numTargetSamples, seen_flurry_keys, startIndex=0):
    possFlurry =[]
    delta = 25
    flurryHead = 0
    targetEpochSize = 50
    targetEpoch = []
    isFlurry = False
    buildCount = 0
    targetMsgCount = 0
    targetStartPositions = []
    j = startIndex 
    threshold = 2
    skip_counter = 0
    # [NEW] adding a set of detected flurry UIDs to prevent overcounting (counting the same flurry twice)
    detectedFlurries = set()
    
    possFlurry = messages[:]

    # want the function to be able to take in numTargetSamples and build the target epochs based on this 
    while j < len(possFlurry) and buildCount < numTargetSamples:
        msg = possFlurry[j]

        if msg['to'] == personOfInterest:
            flurryMessages = []
            allFlurryMessages = []
            #flurryTimes = set()

            for k in range(j, min(j+delta, len(possFlurry))):
                allFlurryMessages.append(possFlurry[k])
                #flurryTimes.add(possFlurry[k]['time'])

                if possFlurry[k]['to'] == personOfInterest:
                    # [NEW] add messages to flurryMessages
                    flurryMessages.append(possFlurry[k])
            
                    #if len(flurryMessages) == threshold:
                        # [NEW] extract unique message IDs in detected flurry
                        #flurryTuple = tuple(sorted((m['time'], m['from'], m['to'], m['UID']) for m in allFlurryMessages))
                        #flurryUIDs = tuple(sorted(m['UID'] for m in flurryMessages))
                        #flurryTimeKey = tuple(sorted(flurryTimes))

                        # [NEW] if there is some overlap, we will skip 
                    #    if flurryTuple in detectedFlurries:
                    #        print("UID OVERLAP -- SKIPPING DUPLICATE FLURRY")
                    #        break



                        # [NEW] else, add the flurry to the detected list 
                        #detectedFlurries.add(flurryTuple)

                    if len(flurryMessages) == threshold:
                        flurryKey = tuple(sorted((m['time'], m['UID']) for m in allFlurryMessages))

                        if flurryKey in seen_flurry_keys:
                            skip_counter += 1
                            print(f"UID OVERLAP -- SKIPPING DUPLICATE FLURRY. SKIP COUNT {skip_counter}")
                            break

                        seen_flurry_keys.add(flurryKey)


                        print(f"FLURRY LOOKS LIKE: {allFlurryMessages}")
                        isFlurry = True
                        flurryHead = j
                        targetStartPositions.append(j)
                        j += delta
                        buildCount += 1
                        break

            j += 1

            # on flurry found, build target epoch
            if isFlurry:
                isFlurry = False # reset
                possTargetEpoch = messages[:flurryHead][::-1]
                targetEpoch = possTargetEpoch[:targetEpochSize]
                #print(f"TARGET EPOCH SIZE: {len(targetEpoch)}")
                #print(f"TARGET EPOCH LOOKS LIKE: {targetEpoch}")
        else:
            j += 1
    print(f"The flurry head is: {flurryHead}")
    print(f"Total size of target epochs is: {len(targetEpoch)} with starting positions: {targetStartPositions}")

    return targetEpoch, j




# to match our generated target epoch, need to get a random epoch as well, 
# it should be the same size as the target epoch
# TODO: need new range for random
def buildRandomEpoch(messages, numTargetSamples):
    randomEpochSize = 50
    #randomEpochSize = 100
    randomEpoch = []
    randomStartPositions = []
    for j in range(numTargetSamples):
        startPos = np.random.choice(len(messages) - randomEpochSize)
        randomStartPositions.append(startPos)
        for i in range(randomEpochSize):
            randomEpoch.append(messages[startPos +  i])
    #print("RANDOM EPOCH SIZE: " + str(len(randomEpoch)) + " WITH STARTING POSITIONS " + str(randomStartPositions))
    #print(f"RANDOM EPOCH LOOKS LIKE: {randomEpoch}")
    return randomEpoch

def attack(messages, personOfInterest, numEpochsAtt, useMartiny=False):
    startIndex = 0
    def check_for_errors(messages, personOfInterest, time):
        if not useMartiny and detectFlurry(messages, personOfInterest) == 0:
            print("No flurries detected, attack aborted and returning zeroed out log.")
            return None
        time = findTimeOfFirstMessageAfterTime(messages, personOfInterest, time)
        if time is None or not isinstance(time, int):
            print("BROKEN ON TIME ERROR.")
            return None
        return time
    
    def sample_epochs(messages, personOfInterest, all_target_messages, all_random_messages, seen_flurry_keys, startIndex):
        if useMartiny:
            new_target_messages, updatedIndex = buildMartinyTargetEpoch(messages, personOfInterest, 1, seen_trigger_indices, startIndex)
            print(f"Started at index: {startIndex}")
            startIndex = updatedIndex
            print(f"Updated index to: {updatedIndex}")
        else:
            new_target_messages, updatedIndex = buildTargetEpoch(messages, personOfInterest, 1, seen_flurry_keys, startIndex)
            print(f"Started at index: {startIndex}")
            startIndex = updatedIndex
            print(f"Updated index to: {updatedIndex}")
        new_random_messages = buildRandomEpoch(messages, 1)

        flurry_times_target = sorted(set(msg['time'] for msg in new_target_messages))
        flurry_times_random = sorted(set(msg['time'] for msg in new_random_messages))
        print(f"[EPOCH {epoch}] Target flurry times: {flurry_times_target}")
        print(f"[EPOCH {epoch}] Random flurry times: {flurry_times_random}")

        all_target_messages.extend(new_target_messages)
        all_random_messages.extend(new_random_messages)

        return new_target_messages, new_random_messages, startIndex
    
    def update_counts(targetMessages, randomMessages, targetRecipientCounter):
        targetRecipients = [i['to'] for i in targetMessages]
        randomRecipients = [i['to'] for i in randomMessages]

        for recipient in targetRecipients:
            targetRecipientCounter[recipient] = targetRecipientCounter.get(recipient, 0) + 1

        for recipient in randomRecipients:
            targetRecipientCounter[recipient] = targetRecipientCounter.get(recipient, 0) - 1

        return targetRecipientCounter 
        
    possRecipients = []
    targetRecipientCounter = {recipient: 0 for recipient in range(100000)}
    epoch = 0
    time = -1
    loopCounter = 0

    all_target_messages = []
    all_random_messages = []
    seen_flurry_keys = set()
    seen_trigger_indices = set()

    print("------ NEW ATTACK EXECUTING ------")
    print(f"CURRENT possRecip: {len(possRecipients)}")

    time = check_for_errors(messages, personOfInterest, time)
    if time is None:
        return None
    
    while epoch < numEpochsAtt:
        loopCounter += 1
        print(f"WHILE LOOP ITERATION:  {loopCounter}")

        new_time = check_for_errors(messages, personOfInterest, time)
        if new_time is None:
            print("[WARNING] Skipping this epoch due to time error.")
            break
        time = new_time
        
        epoch += 1
        print(f"Target + Random epoch/flurry: {str(epoch)}")
        print("PASSED TIME ERRORS.")

        print(f"Calling sample_epochs with startIndex: {startIndex}")
        targetMessages, randomMessages, startIndex = sample_epochs(messages, personOfInterest, all_target_messages, all_random_messages, seen_flurry_keys, startIndex)
        targetRecipientCounter = update_counts(targetMessages, randomMessages, targetRecipientCounter)

        # Adding UID/time-based flurry identifiers to seen_flurry_keys
        for msg in targetMessages:
            seen_flurry_keys.add((msg['time'], msg['UID']))

        targetUIDs = [i['UID'] for i in all_target_messages]
        randomUIDs = [j['UID'] for j in all_random_messages]
        UIDintersection = set(targetUIDs).intersection(set(randomUIDs))
        print(f"For a total of {numEpochsAtt} target & random epochs sampled ({numEpochsAtt * 50} messages), we have {len(UIDintersection)} many common UIDs")
        
    keys_list = list(targetRecipientCounter.keys())
    random.shuffle(keys_list)
    randomized_targetRecipientCounter = {key: targetRecipientCounter[key] for key in keys_list}
    sorted_targetRecipientCounter = sorted(randomized_targetRecipientCounter.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Over a total of {numEpochsAtt} observed flurries, with target {personOfInterest}, we have:")
    for i, (key, value) in enumerate(sorted_targetRecipientCounter[:15]):
        print(f"Entry #{i + 1} ---- ID: {key}, Count: {value}")
    
    print(f"From a total of {len(sorted_targetRecipientCounter)} senders/receivers identified")
    print(sorted_targetRecipientCounter[:30])
    
    return sorted_targetRecipientCounter if sorted_targetRecipientCounter else -1

def martiny_attack(messages, personOfInterest, numEpochsAtt):
    startIndex = 0
    target_epochs_built = 0
    random_epochs_built = 0
    targetRecipientCounter = {recipient: 0 for recipient in range(100000)}
    all_target_messages = []
    all_random_messages = []
    seen_trigger_indices = set()
    seen_flurry_keys = set()

    def update_counts(targetMessages, randomMessages, targetRecipientCounter):
        targetRecipients = [i['to'] for i in targetMessages]
        randomRecipients = [i['to'] for i in randomMessages]

        for recipient in targetRecipients:
            targetRecipientCounter[recipient] = targetRecipientCounter.get(recipient, 0) + 1

        for recipient in randomRecipients:
            targetRecipientCounter[recipient] = targetRecipientCounter.get(recipient, 0) - 1

        return targetRecipientCounter 

    while target_epochs_built < numEpochsAtt and random_epochs_built < numEpochsAtt:
        print(f"WHILE LOOP ITERATION: target={target_epochs_built}, random={random_epochs_built}")

        # Build target epoch
        targetMessages, startIndex = buildMartinyTargetEpoch(messages, personOfInterest, 1, seen_trigger_indices, startIndex)
        if targetMessages and len(targetMessages) == 50:
            all_target_messages.extend(targetMessages)
            target_epochs_built += 1

            # Build random epoch
            randomMessages = buildRandomEpoch(messages, 1)
            if randomMessages and len(randomMessages) == 50:
                all_random_messages.extend(randomMessages)
                random_epochs_built += 1
            else:
                print(f"[WARNING] Failed to build a full random epoch. Skipping increment.")
                break
        else:
            print(f"[WARNING] Failed to build a full target epoch at index {startIndex}. Skipping increment.")
            startIndex += 1
        if startIndex >= len(messages) - 100:
            print("[FATAL] Reached end of available messages. Breaking attack early.")
            break

        targetRecipientCounter = update_counts(targetMessages, randomMessages, targetRecipientCounter)

        # Adding UID/time-based flurry identifiers to seen_flurry_keys
        for msg in targetMessages:
            seen_flurry_keys.add((msg['time'], msg['UID']))

        targetUIDs = [i['UID'] for i in all_target_messages]
        randomUIDs = [j['UID'] for j in all_random_messages]
        UIDintersection = set(targetUIDs).intersection(set(randomUIDs))
        print(f"For a total of {numEpochsAtt} target & random epochs sampled ({numEpochsAtt * 50} messages), we have {len(UIDintersection)} many common UIDs")
        
    keys_list = list(targetRecipientCounter.keys())
    random.shuffle(keys_list)
    randomized_targetRecipientCounter = {key: targetRecipientCounter[key] for key in keys_list}
    sorted_targetRecipientCounter = sorted(randomized_targetRecipientCounter.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Over a total of {numEpochsAtt} observed flurries, with target {personOfInterest}, we have:")
    for i, (key, value) in enumerate(sorted_targetRecipientCounter[:15]):
        print(f"Entry #{i + 1} ---- ID: {key}, Count: {value}")
    
    print(f"From a total of {len(sorted_targetRecipientCounter)} senders/receivers identified")
    print(sorted_targetRecipientCounter[:30])

    return sorted_targetRecipientCounter if sorted_targetRecipientCounter else -1


def buildMartinyTargetEpoch(messages, personOfInterest, numTargetSamples, seen_trigger_indices, startIndex = 0):
    targetEpochSize = 50
    buildCount = 0
    targetEpochs = []
    trigger_indices = []

    j = startIndex
    while j < len(messages) and buildCount < numTargetSamples:
        msg = messages[j]

        if msg['to'] == personOfInterest and j not in seen_trigger_indices:
            print(f"Triggering Martiny Message: {msg}")
            if j+1+targetEpochSize < len(messages):
                epoch = messages[j+1 : j+1+targetEpochSize]
                targetEpochs.extend(epoch)
                seen_trigger_indices.add(j)
                trigger_indices.append(j)
                buildCount += 1
                j += targetEpochSize # skip
        else:
            j += 1 # keep skimming
    print(f"Built {buildCount} many target epochs starting at positions {trigger_indices}")
    print(f"Martiny target epoch looks like: {targetEpochs}")
    return targetEpochs, j

def show_messages_to_from_zero(messages):
    """Show all messages to/from user 0, sorted by UID."""
    relevant_msgs = [msg for msg in messages if msg['to'] == 0 or msg['from'] == 0]
    to_user_0 = [msg for msg in messages if msg['to'] == 0]
    relevant_msgs_sorted = sorted(relevant_msgs, key=lambda x: x['UID'])

    print(f"Total messages involving user 0: {len(relevant_msgs_sorted)}\n")
    print(f"Total messages sent to user 0: {len(to_user_0)}\n")
    for msg in relevant_msgs_sorted:
        print(f"UID: {msg['UID']}, Time: {msg['time']}, Type: {msg['type']}, From: {msg['from']} -> To: {msg['to']}")

    return relevant_msgs_sorted
    
# Want to find the max rank of all group members after the attack has been executed
# to do this, feed the function the output of the attack along with a list of KNOWN group members
# want to get the higest rank among all group members and graph this along with the #epochs observed
# gist: if the highest rank among all members is #GroupMembers, then the group has been fully deanonymized

# SIMULATOR FAULT: getMaxRank has been reviewed. To me it looks fine. 
def getMaxRank(attackResults, targets):
    rankArr = []
    maxRank = 0
    # init
    if attackResults is None:
        print("ERROR: attack results not present")
        return -1
    else:
        pass
    for index, pair in enumerate(attackResults):
        if pair[0] in targets:
            rankArr.append(index)
    # no error
    if len(rankArr) > 0:
        maxRank = max(rankArr)+1
        print("The max rank for the group is: " + str(maxRank))
        return maxRank
    else:
        print("ERROR: could not identify ranks of targets")
        return -1

def getMinRank(attackResults, targets):
    rankArr = []
    minRank = 0
    # init
    if attackResults is None:
        print("ERROR: attack results not present")
        return -1
    else:
        pass
    for index, pair in enumerate(attackResults):
        if pair[0] in targets:
            rankArr.append(index)
    # no error
    if len(rankArr) > 0:
        minRank = min(rankArr) + 1
        print("The min rank for the group is: " + str(minRank))
        return minRank
    else:
        print("ERROR: could not identify ranks of targets")
        return -1

def main():
    with open('fullLogStandard.json','r') as f:
        LogStandard = json.loads(f.read())
    attack(LogStandard, 0)

#def main():
#    with open('log.json','r') as f:
#        log = json.loads(f.read())
#    attack(log, 0)

if __name__== "__main__":
    main()