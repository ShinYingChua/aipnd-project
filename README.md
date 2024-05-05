# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Executing the pipeline
1) train.py
   - Basic usage: `python train.py data_directory`
   - Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
   - Choose architecture: `python train.py data_dir --arch "vgg13"`
   - Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
   - Use GPU for training: `python train.py data_dir --gpu`
2) predict.py
   - Basic usage: `python predict.py /path/to/image checkpoint`
   - Return top K most likely classes: `python predict.py input checkpoint --top_k 3`
   - Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
   - Use GPU for inference: `python predict.py input checkpoint --gpu`

## Results
The predict.py script allows users to print out the top K classes along with associated probabilities and load a JSON file that maps the class values to other category names. <br>
img_path: ./flowers/test/58/image_02743.jpg <br>
top_k: 5 <br>
![image](https://github.com/ShinYingChua/aipnd-project/assets/101923627/f266d3a9-1844-4d06-8cdd-3051a01caa31)
