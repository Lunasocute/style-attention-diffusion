import torch

class Config:
    # Routes configuration
    METADATA_CSV = "data/metadata.csv"          # including file_path, artist, genre, style
    RAW_DATA_PATH = "./data/raw_images"         # image folder
    PROCESSED_DATA_PATH = "./data/processed"
    PAIRS_FILE = "./data/processed/style_pairs.csv"
    TEST_REFS_FILE = "data/test_refs.csv"       # test reference list 
    OUTPUT_DIR = "./checkpoints"
    
    
    # Phase 1：Data parameter
    CONTENT_SIMILARITY_THRESHOLD = 0.8
    TEST_REF_FRACTION_PER_GROUP = 0.05          # proportion for test images of each group
    MIN_IMAGES_PER_GROUP = 7                    # minimum threshold for each group
    
    
    # Phase 2: Model parameter
    SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
    CLIP_IMAGE_MODEL = "openai/clip-vit-large-patch14"
    NUM_STYLE_TOKENS = 4

    # Phase 3: Training parameter
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 1
    STYLE_DROPOUT_PROB = 0.15 
    SELF_STYLE_PROB = 0.9
    STYLE_LOSS_WEIGHT = 0.01
    STYLE_LOSS_INTERVAL = 4
    EVAL_INTERVAL = 250
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"