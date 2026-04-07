# configuration for the models
import yaml


class Config:
    def __init__(self, config_path):

        with open(config_path, encoding='utf-8') as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

        # ----------- parse yaml ---------------#
        self.DATA_PATH = yaml_dict['DATA_PATH']
        self.DATASET = yaml_dict['DATASET']
        self.MODALITY_LIST = yaml_dict['MODALITY_LIST']
        self.IMPUTE_LIST = yaml_dict['IMPUTE_LIST']
        self.IMPUTE_VAL_LIST = yaml_dict['IMPUTE_VAL_LIST']
        self.MODEL_DIR = yaml_dict['MODEL_DIR']

        self.VISUALIZE = yaml_dict['VISUALIZE']
        self.TEST_SAVE = yaml_dict['TEST_SAVE']
        
        # Training parameters
        self.NUM_WORKERS = 2 # when ddp training, pls lower the num_workers.
        self.RANDOM_SEED = 1337

        self.INPUT_C = yaml_dict['INPUT_C']
        self.OUTPUT_C = yaml_dict['OUTPUT_C']

if __name__ == '__main__':
    cfg = Config(config_path='./params.yaml')
    print(cfg)