# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 17:07:06 2018

@author: PC
"""

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from Code import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()