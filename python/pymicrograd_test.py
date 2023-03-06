#!/usr/bin/env python3
from pymicrograd import Value

x = Value(-4.0)
y= x * 2
y.backward()
print(y)
