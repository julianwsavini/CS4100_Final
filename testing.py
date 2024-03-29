import pickle

with open('dataset.pkl', 'rb') as file:
    dataset:dict = pickle.load(file)

# List some attributes about pieces in the dataset
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




