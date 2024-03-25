import pickle

with open('dataset.pkl', 'rb') as file:
    dataset:dict = pickle.load(file)

# List some attributes about peices in the dataset
"""
for piece_name in list(dataset.keys())[:2]:
    piece = dataset[piece_name]
    nmat = piece['nmat']
    root = piece['root']
    mode = piece['mode']
    tonic = piece['tonic']
    print('---------------------------------')
    print(f'piece_name={piece}')
    print(f'mode={mode}')
    print(f'tonic={tonic}')
    print(f'nmat={nmat}')
    print(f'root={root}')
"""
# Attempt to find overlapping notes!
#create a new list with only pieces that don't have overlaps in them
no_overlaps = list()
for piece_name in list(dataset.keys()):
    piece = dataset[piece_name]
    nmat = piece['nmat']
    root = piece['root']
    mode = piece['mode']
    tonic = piece['tonic']
    found_overlap = False  # Flag to indicate overlap
    for i in range(1, len(nmat)):
        prev = nmat[i-1]
        curr = nmat[i]
        same_time_frame = prev[0] == curr[0] and prev[1] == curr[1]  # True if they're at the same time
        subsequent = prev[1] <= curr[0]  # True if they don't overlap
        if not (subsequent or same_time_frame):
            found_overlap = True
            break
    if not found_overlap:
        no_overlaps.append((nmat, root, mode, tonic))

#function for later, takes out the last note/group of notes if there are several at the same time
#for the model to predict it given the original sequence
#this function assumes no note overlaps
#returns a tuple (array of last note group, array of notes beforehand)
def separate_last_note_group(nmat):
    x = len(nmat)-1 #index for last note
    last_group =[]
    for i in range(x, -1, -1):
        curr = nmat[i]
        if curr[0] != nmat[x][0]:
            break
        last_group.append(curr)
    last_group.reverse()
    return last_group, nmat[:-(len(last_group))]

#testing
"""
piece = no_overlaps[1100][0]
print(f'original: {piece}')
last_group, all_but_last = separate_last_note_group(piece)
print(f'last note group: {last_group}, all but last: {all_but_last}')
"""

