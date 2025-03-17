import json
import numpy as np
import random

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
                            print(f"FLURRY IDENTIFIED, INC numFlurries to {numFlurries}")
                        j += delta
                        break
        j += 1
        targetMsgCount = 0
    print(f"numFlurries: {numFlurries}")   
    return numFlurries



## is the look ahead window built correctly?

# graphs for differing epoch lengths 
def buildTargetEpoch(messages, personOfInterest, numTargetSamples):
    possFlurry =[]
    delta = 25
    flurryHead = 0
    targetEpochSize = 50
    targetEpoch = []
    isFlurry = False
    buildCount = 0
    targetMsgCount = 0
    targetStartPositions = []
    j = 0 
    threshold = 2
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
            
                    if len(flurryMessages) == threshold:
                        # [NEW] extract unique message IDs in detected flurry
                        flurryTuple = tuple(sorted((m['time'], m['from'], m['to'], m['UID']) for m in allFlurryMessages))
                        #flurryUIDs = tuple(sorted(m['UID'] for m in flurryMessages))
                        #flurryTimeKey = tuple(sorted(flurryTimes))

                        # [NEW] if there is some overlap, we will skip 
                        if flurryTuple in detectedFlurries:
                            print("UID OVERLAP -- SKIPPING DUPLICATE FLURRY")
                            break

                        # [NEW] else, add the flurry to the detected list 
                        detectedFlurries.add(flurryTuple)

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
                print(f"TARGET EPOCH SIZE: {len(targetEpoch)}")
                print(f"TARGET EPOCH LOOKS LIKE: {targetEpoch}")
        else:
            j += 1
    print(f"The flurry head is: {flurryHead}")
    print(f"Total size of target epochs is: {len(targetEpoch)} with starting positions: {targetStartPositions}")

    return targetEpoch




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
    print("RANDOM EPOCH SIZE: " + str(len(randomEpoch)) + " WITH STARTING POSITIONS " + str(randomStartPositions))
    return randomEpoch





# attack functionality here
def attack(messages, personOfInterest, numEpochsAtt):
    possRecipients = []
    targetRecipientCounter = {}
    list_targetRecipientCounter = []
    epoch = 0
    time = -1
    loopCounter = 0
    noFlurry = 0
    initCount = 1
    targetUIDs = []
    randomUIDs = []
    UIDintersection = 0
    # initialize the count for all users to be zero. 
    # This will fix problems with the maxRank() function later when target users do not produce flurries:
    for recipient in range(10000):
        targetRecipientCounter[recipient] = 0

    print("------ NEW ATTACK EXECUTING ------")
    print("CURRENT possRecip: " + str(len(possRecipients)))

    # while condition should be refactored for new attack functioning
    while epoch < numEpochsAtt and True:
        loopCounter += 1
        print("WHILE LOOP ITERATION: " + str(loopCounter))
        time = findTimeOfFirstMessageAfterTime(messages, personOfInterest, time)
        print("TIME: " + str(time))
        print("time type is: " + str(type(time)))
        if time is not None:
            if not int(time) == time:
                print("BROKEN ON TIME ERROR.")
                break
            else:
                pass
        else:
            print("BROKEN ON TIME ERROR.")
            break
        # here I think we no longer want to break, but rather iterate some counter to break on...? 
        # or should we just loop through everything? 
        if detectFlurry(messages, personOfInterest) == 0:
            noFlurry += 1
            if noFlurry >= 5:
                print("BROKEN ON FLURRIES DNE.")
                break
        epoch += 1
        print("target+random epoch/flurry: " + str(epoch))
        print("PASSED FLURRY AND TIME ERRORS.")
        # BIT 1
        # grabs target epoch at applicable time (is time needed?)
        # done every run
        targetMessages = buildTargetEpoch(messages, personOfInterest, numEpochsAtt)
        print("TARGET MESSAGES: " + str(targetMessages))
        # grabs the recipients
        targetRecipients = [i['to'] for i in targetMessages]
        print("targetMessages SIZE: " + str(len(targetMessages)))
        print("targetRecipients SIZE: " + str(len(targetRecipients)))
        # iterate over all recipients, want to keep an associated count for them
        # to track their appearances in target epochs
        # want a dictionary of users and their associated counts to be updated on attack runs
        # e.g. {'user' : count} -- {16 : 24} ... i think (?)
        for recipient in targetRecipients:
            if recipient in targetRecipientCounter:
                # recipient found in target epoch
                targetRecipientCounter[recipient] += 1
                if recipient in range(5): 
                    print("RECIPIENT " + str(recipient) + " IN DICTIONARY! -- INC COUNT")
            else:
                # not in, they need to be added -- 
                # note: on first pass this will add everyone (in targetRecipients) with a initialized count of one
                targetRecipientCounter[recipient] =  initCount
                if recipient in range(5): 
                    print("RECIPIENT "  + str(recipient) + " NOT FOUND IN DICTIONARY! -- ADDING NOW")

        #BIT 2 -- needs to be done numFlurries many times
        #randomSamples = detectFlurry(messages, personOfInterest)
        randomMessages = buildRandomEpoch(messages, numEpochsAtt)
        print("LENGTH OF RANDOM EPOCH(S): " + str(len(randomMessages)))
        # grab recipients 
        randomRecipients = [i['to'] for i in randomMessages]
        #print("RANDOM EPOCH: " + str(randomMessages))
        print("randomMessages SIZE: " + str(len(randomMessages)))
        # iterate over all recipients, check if they are members of targetRecipients
        # if they are, we need to decrement their count
        # if NOT, do nothing
        for recipient in randomRecipients:
            if (recipient in targetRecipientCounter):
                # this is a popular user, decrement their count
                targetRecipientCounter[recipient] -= 1 
                if recipient in {0, 1, 2, 3, 4}: 
                    print("RECIPIENT "  + str(recipient) + " FOUND IN RANDOM EPOCH! -- DEC COUNT")
            else:
                targetRecipientCounter[recipient] = -1

        # want to check the number of intersecting UIDs in both TARGET and RANDOM epochs
        targetUIDs = [i['UID'] for i in targetMessages]
        randomUIDs = [j['UID'] for j  in randomMessages]
        targetSet = set(targetUIDs)
        print(targetSet)
        randomSet = set(randomUIDs)
        print(randomSet)
        UIDintersection = targetSet.intersection(randomSet)
        print("For a total of " + str(numEpochsAtt) + " target & random epochs sampled(" + str(numEpochsAtt * 50) + " messages), we have a total of " + str(len(UIDintersection)) + " common UIDs")
        # at the end, we want to return the target recipient counter,
        # sorted in descending order so that the most popular IDs are at the top
        # the likely group members are those at the top with similar/identical counts
        # first, randomize order of the counter
        keys_list = list(targetRecipientCounter.keys())
        random.shuffle(keys_list)
        randomized_targetRecipientCounter = {key: targetRecipientCounter[key] for key in keys_list}
        # now sort it
        sorted_targetRecipientCounter = sorted(randomized_targetRecipientCounter.items(), key=lambda x: x[1], reverse=True)
        miscCounter = 0
        print("Over a total of " + str(numEpochsAtt) + " observed flurries, with target " + str(personOfInterest) + ",  we have: ")
        if sorted_targetRecipientCounter:
            for key, value in sorted_targetRecipientCounter:
                if miscCounter < 15:
                    miscCounter += 1
                    print("Entry #" + str(miscCounter) + " ---- " "ID: " + str(key) + ", " + "Count: " + str(value))
            print("From a total of " + str(len(sorted_targetRecipientCounter)) + " senders/receivers identified")
            print(sorted_targetRecipientCounter[0:30])
            return sorted_targetRecipientCounter
            #groupMaxRank = getMaxRank(sorted_targetRecipientCounter, (0,1,2))
            #return groupMaxRank
        else: 
            print("No senders found. -- possible error")
            return -1
        
# Want to find the combined rank of all group members after the attack has been executed
# to do this, feed the function the output of the attack along with a list of KNOWN group members
# want to get the sum of their ranks and graph this along with the #Epochs observed to get an idea
# of how fast the attack can function
def getCombinedRank(attackResults, targets):
    combinedRank = 0
    # initialize
    for index, pair in enumerate(attackResults):
        if pair[0] in targets:
            combinedRank += index
    # no error
    if combinedRank > 0:
        print("The combined rank of " + str(targets) +  " is: " +str(combinedRank))
        return combinedRank
    else:
        # error
        return -1
    
# Want to find the max rank of all group members after the attack has been executed
# to do this, feed the function the output of the attack along with a list of KNOWN group members
# want to get the higest rank among all group members and graph this along with the #epochs observed
# gist: if the highest rank among all members is #GroupMembers, then the group has been fully deanonymized

# SIMULATOR FAULT: getMaxRank has been reviewed. To me it looks fine. 
def getMaxRank(attackResults, targets):
    rankArr = []
    maxRank = 0
    # init
    if type(attackResults) is None:
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
