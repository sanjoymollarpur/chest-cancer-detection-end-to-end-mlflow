from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier import logger
import os
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/sanjoymollarpur/chest-cancer-detection-end-to-end-mlflow.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="sanjoymollarpur"
os.environ["MLFLOW_TRACKING_PASSWORD"]="8030f3bc3f62e76900ae582df3bb6047a371d1a4"


STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        # evaluation.save_score()
        # logger.info(f"Model evalution-------0")
        
        evaluation.log_into_mlflow()
        # logger.info(f"Model evalution-------1")




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
            