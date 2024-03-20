import pickle

with open('dataset.pkl', 'rb') as file:
    dataset:dict = pickle.load(file)

# List some attributes about peices in the dataset
for piece_name in list(dataset.keys())[:2]:
    piece = dataset[piece_name]
    nmat = piece['nmat']
    root = piece['root']
    mode = piece['mode']
    tonic = piece['tonic']
    print('---------------------------------')
    print(f'{piece_name=}')
    print(f'{mode=}')
    print(f'{tonic=}')
    print(f'{nmat=}')
    print(f'{root=}')

# Attempt to find overlapping notes!
# for piece_name in list(dataset.keys()):
#     piece = dataset[piece_name]
#     nmat = piece['nmat']
#     for i in range(1,len(nmat)):
#         prev = nmat[i-1]
#         curr = nmat[i]
#         same_time_frame = prev[0] == curr[0] and prev[1] == curr[1]
#         subsequent = prev[1] == curr[0]
#         if not (subsequent or same_time_frame): print(f'found overlap: {prev=},\n {curr=}')
