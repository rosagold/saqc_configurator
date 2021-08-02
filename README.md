SaQC Configurator
=================

Parameter
---------
pass different types just like python code.
- `None`, `inf`, `-inf`, `nan` are the only known constants
- `str`: quote the value with `"` or `'`, eg. `"foo"`
- `float`: to force integers to floats use a `.0` or just the `.`, eg `4.`
- `int`: just pass a number
- `list`: eg. `[None, 1, 3.3, nan]` 
- `dict`: eg. `{"a": 11, None: None}`
- `tuple`: eg. `(3,)`, `(1,1)`