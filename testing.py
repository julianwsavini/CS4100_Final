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


