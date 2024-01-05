import gdown

url="https://drive.google.com/file/d/1sQu28HTUMBsFy_tof3iEopcrk8wvbLXf/view?usp=sharing"

file_id=url.split("/")[-2]

print(file_id)

prefix="https://drive.google.com/uc?/export=download&id="

gdown.download(prefix+file_id, "chest_ct.zip")

