import random

# data structures
class HashStructure:
    key = None 

class HashSorted:
    hashSet = set() # interface HashStructure
    data = []  # interface of HashStructure

    def __init__(self, use_key=False):
        self.hashSet = set()
        self.data = []
        self.use_key = use_key

    def contains(self, object)->bool:
        # O(1)
        if self.use_key:
            return object.key in self.hashSet
        return object in self.hashSet

    def getRandomElement(self):
        # O(1)
        len_hash = self.size()
        if len_hash > 0:
            return self.data[random.randint(0,len_hash-1)]
        return None

    def size(self):
        # O(1)
        return len(self.data)

    def add(self,object:HashStructure):
        # O(1)
        if self.contains(object):
            return
        self.data.append(object)
        if self.use_key:
            self.hashSet.add(object.key)
        else:
            self.hashSet.add(object)


        

    def addSorted(self,object:HashStructure):
        # todo: improve to O(log n). Actual O(n)
        if object in self.hashSet:
            return

        len_data:int = self.size()
        if len_data == 0:
            self.add(object)
        else:
            inserted:bool = False
            for i in range(len_data):
                key:float = self.data[i].key

                if object.key <= key:

                    self.data.insert(i, object)

                    if self.use_key:
                        self.hashSet.add(object.key)
                    else:
                        self.hashSet.add(object)
                    inserted = True
                    break
            if not inserted:
                self.add(object)


    def clear(self):
        # O(1)
        self.hashSet.clear()
        self.data.clear()

    def get(self,index:int):
        # O(1)
        if index < 0 or index >= self.size():
            return None
        return self.data[index]
        
    def removeByIndex(self,index:int):
        # O(1)
        if index < 0 or index >= self.size():
            return None
        self.hashSet.remove(self.data[index])
        self.data.pop(index)
        
    def remove(self,object):
        # O(1)
        if self.use_key:
            self.hashSet.remove(object.key)
        else:
            self.hashSet.remove(object)
        self.data.remove(object)