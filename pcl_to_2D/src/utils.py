#!/usr/bin/env python

import struct
import numpy as np

def cloud_to_numpy(cloud):
	return np.array([(x, y, z, 1.0) for rgb, x, y, z, i in cloud])
