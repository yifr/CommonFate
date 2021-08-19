import pickle
import sys

path = sys.argv[1]
with open(path, "rb") as f:
    config = pickle.load(f)

for obj in config['objects'].keys():
    obj_conf = config['objects'][obj]
    print(obj)
    print('\t', obj_conf['shape_type'])
    print('\t', obj_conf['shape_params'])
    print('\t', obj_conf['scaling_params'])
