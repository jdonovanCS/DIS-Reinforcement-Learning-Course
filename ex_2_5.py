# import numpy as np
# import maypolotlib.pyplot as plt

# Q = []
# N = []

# # create our bandits with an initial 
# def initializeBandits(k, mean=0, std=.01, b_std=.01):
    
#     bandits = []

#     for i in range(1, k):
#         bandit.append([np.random(mean, std)], b_std)


import numpy as np
import matplotlib.pyplot as plt


#create our bandits with initial values for reward and a std for that reward
def create_bandits(n, mean=0, std=1, b_std=1):
    bandits = []

    for i in range(n):
        bandits.append([np.random.normal(mean, std), b_std])

    return bandits


def bandit_random_walk(bandits, shift):
    for bandit in bandits:
        bandit[0] += shift if np.random.uniform() < 0.5 else -shift
    return bandits


def main():

    num_bandits = 10
    
    #amount of episodes for convergence - what is the difference in reruns vs episodes from logical perspective
    episodes = 1000

    #step size parameter
    alpha = 0.1

    #exploration chance parameter
    epsilon = 0.1

    #random walk shift for each bandit's reward over time.
    shift = 0.01

    #What is this?
    merge_choices_num = 500

    #amount of reruns for convergence
    reruns = 1000

    merged_choices = []

    #rerun to get a better average of the performance
    for h in range(reruns):
        
        #keep track of the number of times we chose the best bandit (made the correct choice)
        correct_choices = []

        #create our bandits
        bandits = create_bandits(num_bandits)

        max_bandit = 0
        best_bandit = 0
        for i in range(len(bandits)):
            
            #set max bandit to the reward of our best bandit
            #set best bandit to the number of the best bandit
            if bandits[i][0] > max_bandit:
                max_bandit = bandits[i][0]
                best_bandit = i
                #print str(max_bandit)

        #instantiate q and n for all bandits to zero
        #q being the estimated value for that bandit (?)
        #n being the number of times we've chosen that bandit.
        q = np.zeros(num_bandits)
        n = np.zeros(num_bandits)

        # repeat for a number of episodes to get a better average for the values for the bandits.
        for i in range(episodes):
            
            #shift bandit randomly for better or worse by shift amount
            bandit_random_walk(bandits, shift)    # Random walk by shift amount

            #select the bandit with the best estimated value
            selected_bandit = np.argmax(q)

            #decide if we will keep selected bandit with best estimated value (greedy) or if we will explore a random other bandit
            if np.random.uniform() < epsilon:
                selected_bandit = np.random.choice(len(q))

            #get the actual reward based on the reward with a std of b_std from bandit creation function
            reward = np.random.normal(bandits[selected_bandit][0], bandits[selected_bandit][1])

            #increment n (count) for this bandit
            n[selected_bandit] += 1

            ##Update the estimated value for the future for this bandit.
            q[selected_bandit] += (reward - q[selected_bandit]) / n[selected_bandit]    # Sample Average
            #alpha = 1/(i + 1)  # Variable alpha
            #q[selected_bandit] += alpha * (reward - q[selected_bandit])   # Using alpha

            #if we selected the bandit with the highest actual reward, then append a 1 (true) to this array
            #else append a 0 (false) to this array.
            correct_choices.append(float(int(selected_bandit == best_bandit)))

            #if we are on an episode before merge_choices_num, then merged_choice = correct choice percentage
            if i <= merge_choices_num:
                merged_choice = sum(correct_choices)/len(correct_choices)
            #else 
            else:
                merged_choice = sum(correct_choices[-merge_choices_num:])/merge_choices_num
            #print str(merged_choice)

            # if this is the first run, append the merged_choice into merged_choices
            if h == 0:
                merged_choices.append(merged_choice)
            # else update the merged_choice for this episode with the 
            else:
                merged_choices[i] += (merged_choice - merged_choices[i]) / (h + 1)

            #print str(merged_choices[i])

    plt.axis([0, episodes, -0.5, 1.5])
    plt.plot(range(episodes), merged_choices)
    plt.show()


if __name__ == "__main__":
    main()
        
