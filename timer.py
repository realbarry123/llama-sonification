import time

class Timer():
    def __init__(self):
        self.start_time = time.time()
    
    def reset(self):
        self.__init__(self)

    def log(self, message: str): 
        print(f"{time.time()-self.start_time}:", message)