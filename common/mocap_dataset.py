# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

class MocapDataset:
    def __init__(self):
        self._data = None # Must be filled by subclass
        self._cameras = None # Must be filled by subclass
    
    def __getitem__(self, key):
        return self._data[key]
        
    def subjects(self):
        return self._data.keys()
        
    def cameras(self):
        return self._cameras
