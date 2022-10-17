

cdef class DummyClass:
    cdef double a
    
    def __init__(self, double inputVar):
        self.a = inputVar
    
    cpdef double doubleVar(self):
        return self.a * 2

def printHi():
    print('Hi')