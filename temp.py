import pickle

with open('mc_result.pkl', 'rb') as read:
    scores, win, high = pickle.load(read)

print(sum(high)/100)