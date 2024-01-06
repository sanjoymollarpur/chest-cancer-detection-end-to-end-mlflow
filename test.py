import gdown

url="https://drive.google.com/file/d/1sQu28HTUMBsFy_tof3iEopcrk8wvbLXf/view?usp=sharing"

file_id=url.split("/")[-2]

print(file_id)

prefix="https://drive.google.com/uc?/export=download&id="

gdown.download(prefix+file_id, "chest_ct.zip")



# MLFLOW_TRACKING_URI=https://dagshub.com/sanjoymollarpur/chest-cancer-detection-end-to-end-mlflow.mlflow \
# MLFLOW_TRACKING_USERNAME=sanjoymollarpur \
# MLFLOW_TRACKING_PASSWORD=8030f3bc3f62e76900ae582df3bb6047a371d1a4 \
# python script.py