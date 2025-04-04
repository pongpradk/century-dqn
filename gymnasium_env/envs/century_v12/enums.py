from enum import Enum

class Actions(Enum):
    rest = 0
    getM3 = 1
    getM4 = 2
    getM5 = 3
    getM6 = 4
    getM7 = 5
    getM8 = 6
    getM9 = 7
    getM10 = 8
    useM1 = 9
    useM2 = 10
    useM3 = 11
    useM4 = 12
    useM5 = 13
    useM6 = 14
    useM7 = 15
    useM8 = 16
    useM9 = 17
    useM10 = 18
    getG1 = 19
    getG2 = 20
    getG3 = 21
    getG4 = 22
    getG5 = 23

class CardStatus(Enum):
    NOT_OWNED = 0
    UNPLAYABLE = 1
    PLAYABLE = 2 