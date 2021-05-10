

def get_list_img_to_use():
  text_file=r"data\files_new_experiment.txt"
  with open(text_file) as f:
    lines = f.read().split('\n')[:-1]
  return lines