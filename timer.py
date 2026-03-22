import time

class Timer():
    def __init__(self):
        self.start_time = time.time()
    
    def reset(self):
        self.__init__(self)
    
    def get_time(self):
        return time.time()-self.start_time

    def log(self, message: str): 
        print(f"{time.time()-self.start_time}:", message)