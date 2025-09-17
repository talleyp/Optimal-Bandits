import numpy as np
from bandits import BernoulliBandit

def FollowTheLeader(bandit, n):
    K = bandit.K()  # Access the attribute directly
    
    # Track sum of rewards and pull counts for each arm
    rewards_sum = np.zeros(K)
    pull_counts = np.zeros(K)
    
    # Store the total reward
    total_reward = 0
    
    # Loop over all time steps
    for t in range(n):
        # Initial exploration phase
        if t < K:
            arm_to_pull = t
        else:
            # Exploitation phase: find the current best arm (tie-breaking)
            # Find all arms with the highest average reward
            # This is done by getting the max average and then finding all arms that equal it
            
            # This part handles the initial zero-division problem after the first few pulls
            # where a pull_count could be 0, leading to a NaN.
            current_means = np.divide(rewards_sum, pull_counts, out=np.zeros_like(rewards_sum), where=pull_counts!=0)
            
            max_mean = np.max(current_means)
            best_arms = np.where(current_means == max_mean)[0]
            
            # Randomly break ties
            arm_to_pull = np.random.choice(best_arms)
        
        # Pull the chosen arm
        reward = bandit.pull(arm_to_pull)
        
        # Update our tracking variables
        rewards_sum[arm_to_pull] += reward
        pull_counts[arm_to_pull] += 1
        total_reward += reward
        
    return total_reward

# Example usage
K = 5
N = 100
# Ensure that BernoulliBandit has a .K attribute, not a .K() method
mybandit = BernoulliBandit(np.random.random(K))

total_reward = FollowTheLeader(mybandit, N)

print(f"Total reward from Follow-the-Leader algorithm: {total_reward}")
print(f"Final regret: {mybandit.regret()}")