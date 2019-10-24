# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:49:23 2018

@author: dsher
"""

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run()