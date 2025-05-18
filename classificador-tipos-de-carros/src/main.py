from pathlib import Path
from metadata.data_pipeline import DataPipeline
from preprocessing.preprocessing_pipeline import PreprocessingPipeline

class Main:
    """
    Entry point for running data preprocessing and model training pipeline.

    Args:
        preprocess_data_flag (bool): Flag to run the preprocess pipeline or not.
    """
    def __init__(self, preprocess_data_flag: bool):
        self.preprocess_data_flag = preprocess_data_flag

    def preprocess_metadata(self):
        mat_path = Path('/home/leticia/projetos/Avaliacao-MOBIT/classificador-tipos-de-carros/data/raw/bmw10_annos.mat')
        DataPipeline(mat_path, '/home/leticia/projetos/Avaliacao-MOBIT/classificador-tipos-de-carros/data/raw/bmw10_ims').run()

    def preprocess_images(self):
        metadata_csv_path = '/home/leticia/projetos/Avaliacao-MOBIT/classificador-tipos-de-carros/data/processed/metadata.csv'
        output_dir_path = '/home/leticia/projetos/Avaliacao-MOBIT/classificador-tipos-de-carros/data/processed'
        PreprocessingPipeline(metadata_csv_path, output_dir_path).run()

    def run(self):
        if self.preprocess_data_flag == True:
            #self.preprocess_metadata()
            self.preprocess_images()

if __name__ == "__main__":
    app = Main(True)
    app.run()