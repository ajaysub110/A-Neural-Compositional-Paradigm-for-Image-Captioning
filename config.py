class Config():
    def __init__(self):
        # Hyperparameters:
        # General
        self.EMBEDDING_DIM = 1024
        # Feature/NP Extraction CNN
        self.NP_NUM_CLASSES = 1000    
        # Connecting Module
        self.SEQ_LEN = 10