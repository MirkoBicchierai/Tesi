# Generating 3D facial expressions using an LSTM network
In the requirements.txt file, you will find all the libraries along with their respective versions and dependencies necessary to initiate the training of the network.

You can download the CoMA dataset from here: https://coma.is.tue.mpg.de/, while the CoMA_Florence dataset is available here: https://drive.google.com/drive/folders/14TLFQkWXPwujeApwpjbS15ZYA7_zirwl.

For the CoMA dataset, all 12 labels have been used for each face, whereas for the CoMA_Florence dataset, only 10 labels out of 70 have been used (Cheeky, Confused, Cool, Displeased, Happy, Kissy, Moody, Rage, Sad2, Scream).

# Result with CoMA_Florence

<img align="left" src="https://github.com/MirkoBicchierai/Tesi/blob/master/git_readme_img/CH01_Cheeky.gif" width="50%">
<img align="left" src="https://github.com/MirkoBicchierai/Tesi/blob/master/git_readme_img/CH01_Confused.gif" width="50%">
<img align="left" src="https://github.com/MirkoBicchierai/Tesi/blob/master/git_readme_img/CH01_Cool.gif" width="50%">
<img align="left" src="https://github.com/MirkoBicchierai/Tesi/blob/master/git_readme_img/CH01_Displeased.gif" width="50%">
<img align="left" src="https://github.com/MirkoBicchierai/Tesi/blob/master/git_readme_img/CH01_Happy.gif" width="50%">
<img align="left" src="https://github.com/MirkoBicchierai/Tesi/blob/master/git_readme_img/CH01_Kissy.gif" width="50%">
<img align="left" src="https://github.com/MirkoBicchierai/Tesi/blob/master/git_readme_img/CH01_Moody.gif" width="50%">
<img align="left" src="https://github.com/MirkoBicchierai/Tesi/blob/master/git_readme_img/CH01_Rage.gif" width="50%">
<img align="left" src="https://github.com/MirkoBicchierai/Tesi/blob/master/git_readme_img/CH01_Sad2.gif" width="50%">
<img align="left" src="https://github.com/MirkoBicchierai/Tesi/blob/master/git_readme_img/CH01_Scream.gif" width="50%">

# Setup Folder

Create the "Dataset_FLAME_Aligned" folder and place the CoMA_Florence dataset inside it before extracting the landmarks.

Create the "Dataset_FLAME_Aligned_COMA" folder and perform the same procedure for the CoMA dataset before extracting the landmarks.

# How to use


The CoMA_Florence dataset provides .obj files for each frame that need to be converted to .ply format for proper functioning.

To extract landmarks from the CoMA_Florence dataset, use the file get_animation_landmark_ply.py.

For the sequences in the CoMA dataset, interpolation and downsampling to 40 frames per sequence have been performed. The code for this can be found in the sampling_COMA.py file. Once this operation is completed, you can extract landmarks for each frame using the code in the get_landmark_COMA.py file.

The main.py file contains the training loop for the LSTMCell network for generating 3D expressions, while the test.py file handles the actual generation of the sequence.

In the "Classification" folder, you'll find the LSTM network for classifying the generated sequences.


