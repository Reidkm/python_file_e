# =======================================
# demo for yaml file reading operation
# =======================================

#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import yaml


if __name__ == "__main__":
    cali_config = None
    with open('calibration.yaml','r') as stream:
        try:
            cali_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    print cali_config
    print type(cali_config)
    print cali_config['front_camera']['center_point']
    print type(cali_config['front_camera']['center_point'])  

# =======================================
# yaml file read/write operation
# =======================================

# -*- coding: utf-8 -*-
import yaml
import io

# Define data
data = {'a list': [1, 42, 3.141, 1337, 'help', u'€'],
       'a string': 'bla',
       'another dict': {'foo': 'bar',
                        'key': 'value',
                        'the answer': 42}}

# Write YAML file
with io.open('data.yaml', 'w', encoding='utf8') as outfile:
   yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

# Read YAML file
with open("calibration.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)
    

    
# Write YAML file
with io.open('calibration.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(data_loaded, outfile, default_flow_style=False, allow_unicode=True)
    
#print data_loaded
#print(data == data_loaded)


