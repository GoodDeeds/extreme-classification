#! /usr/bin/env python

import os
import sys
import yaml

if len(sys.argv) != 5:
    print("Usage: python " + sys.argv[0] + " <input-dim> <output-dim> <num-layers> <output-path>")
    sys.exit(1)

input_dims = int(sys.argv[1])
output_dims = int(sys.argv[2])
num_layers = int(sys.argv[3])
output_path = sys.argv[4]

assert input_dims > 0, "Input dimensions must be positive"
assert output_dims > 0, "Output dimensions must be positive"
assert num_layers > 0, "Number of layers must be positive"
assert not os.path.exists(output_path), output_path + " already exists"

out = []

diff_dims = output_dims - input_dims
diff_per_layer = diff_dims / num_layers

curr_dim = input_dims
for layer_idx in range(num_layers):
    layer = {}
    layer['name'] = "Linear"
    if layer_idx != num_layers - 1:
        layer['kwargs'] = {"in_features": curr_dim,
                           "out_features": int(curr_dim + diff_per_layer * (layer_idx + 1))}
    else:
        layer['kwargs'] = {"in_features": curr_dim, "out_features": output_dims}
    out.append(layer)
    if layer_idx != num_layers - 1:
        out.append({"name": "LeakyReLU", "kwargs": {"negative_slope": 0.2, "inplace": True}})
    curr_dim = int(curr_dim + diff_per_layer * (layer_idx + 1))
out.append({"name": "Sigmoid"})

print(yaml.dump(out, default_flow_style=False))

with open(output_path, 'w') as yaml_file:
    yaml.dump(out, yaml_file, default_flow_style=False)
