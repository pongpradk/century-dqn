from enum import Enum

class Actions(Enum):
    rest = 0
    getM2 = 1
    getM3 = 2
    getM4 = 3
    getM5 = 4
    getM6 = 5
    getM7 = 6
    getM8 = 7
    getM9 = 8
    useM1 = 9
    useM2 = 10
    useM3 = 11
    useM4 = 12
    useM5 = 13
    useM6 = 14
    useM7 = 15
    useM8 = 16
    useM9 = 17
    getG1 = 18
    getG2 = 19
    getG3 = 20
    getG4 = 21
    getG5 = 22

class CardStatus(Enum):
    NOT_OWNED = 0
    UNPLAYABLE = 1
    PLAYABLE = 2 