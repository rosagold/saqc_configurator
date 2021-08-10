SaQC Configurator
=================

Requirements
------------
- `docker`
- `docker-compose`

Usage
-----
Build and run docker container local with
```shell
docker-compose up
```
Open a browser and go to
[http://0.0.0.0:8000/saqc-config-app/](http://0.0.0.0:8000/saqc-config-app/)

> **Hint**: To rebuild a docker image after some source files changed use `docker-compose up --build`


Parameter in the GUI
--------------------
To pass different types of **Parametes** in the *Function* section, one pass the values
as in python: 
- `None`, `inf`, `-inf`, `nan` are the only known constants
- `str`: quote the value with `"` or `'`, eg. `"foo"`
- `float`: to force integers to floats use a `.0` or just the `.`, eg `4.`
- `int`: just pass a number
- `list`: eg. `[None, 1, 3.3, nan]` 
- `dict`: eg. `{"a": 11, None: None}`
- `tuple`: eg. `(3,)`, `(1,1)`
- `callables`: **Not supported yet, but soon**


Local installation for developer
--------------------------------

1. clone repro
2. create a virtualenv
3. activate it
4. install requirements
5. run `python wsgi.py`
6. see the console log messages for the address under which the app is run

The app is organized in different files:
- `layout.py`: create a layout for the app.
- `app.py`: create the dash-app and its cache and attach the layout to it, therefore it needs `layout.py`
- `callbacks.py`: define the server-side callbacks and attach them to the app, therefore it needs `app.py`
- `wsgi.py`: the *main*, import the already created app from `callbacks.py` and run it
- `helper.py`: helper functions for `layout.py` and `callback.py`
- `const.py`: constants, constant lists and dictionaries which are needed anywhere in the project

Internals and Buzzwords
-----------------------

The app is written in **python** mainly with the **Dash** package.
It uses the Webframework **Flask** to create a **WSGI**-app, wich then can communicate 
with WSGI-compatible servers. 

On client (Browser) side it uses **Bootstrap** components. On the server side, 
python functions handle the so-called *callbacks*. For communication between the 
callbacks itself, a (Flask-)Filesystem-Cache is used, to store data, that is primary 
used for calculations on server-side.

If the app is run directly via `python wsgi.py` a Flask-server is started in local mode,
which is used for testing/debugging only.
To deploy the app a **docker** container is provided. It set up the app and a 
**gunicorn** server, which communicate with the app via **WSGI** and talk **http** to 
the outside world.