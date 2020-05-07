# Using Deep Learning Techniques for Automatic Short Answer Grading

Bachelor Thesis researching weaknesses of deep learning (DL) models for automatic short answer 
grading (ASAG) with the help of adversarial rules.  

### Prerequisites

The project was created on a PC with Linux Mint 19.2 as OS

### Installing

Create the conda environment with the provided yml file:

```
$ conda env create --file thesis_env.yml
```

### Preprocessing

Before training the ASAG model, we need to preprocess the data sets.

Follow this procedure for SciEntsBank and Beetle 3-way, if the ASAG model is based on BERT.

1. Copy the location of the folder containing the data to preprocess
2. In bert/preprocessing_bert.py, insert the location of the dataset into line 143, variable data_path.
3. Specify where you want to save the preprocessed data in the variable save_path
4. Specify, if you are preprocessing test or training data, by inserting either 'test' or 'train' for the variable mode.
5. Run preprocessing_bert.py

For MNLI, use mnli_bert_preprocessor and insert the file location of the test/training data in step 2 instead (also, 
run mnli_preprocessing_bert.py).

If the ASAG model is based on ALBERT, corresponding files are provided in the folder albert.

### Training 

To train a BERT model, open the training script bert_training.py (albert_training.py) and insert the locations for the 
model, train loss and validation loss, into the lines 19-21 (22-24).
Also, specify which training and validation set to use in line 29-30 (36-37).
Change the hyperparameter settings to your liking and run the training script.

### Find correct guesses

To create a set of correct "incorrect" short answers from the validation set extracted from the training data, open the 
find_correct_guesses.py (find_correct_guesses_albert.py)
and insert the model path in line 16 (15), the path of the validation set in line 17 (16) and specify the path where the
data will be saved in line 18(17).
Then you are able to run the script.

### Testing

To test the performance of a BERT (ALBERT) model,
look into the bert_testing.py (albert_testing.py) scripts, insert the model path and the test set path (line 16-17).
Additionally, specify where the test results should be saved (line 18).
Then the testing script can be run.

### Finding rules 

The rulefinding code can be found here: https://github.com/SebOchs/sears.git

Clone these githubs to your preferred location:

```
1.
git clone https://github.com/SebOchs/sears.git
git clone https://github.com/SebOchs/OpenNMT-py
cd OpenNMT-py/
python setup.py install
cd ..
2.
git clone https://github.com/SebOchs/TextFooler
```

To generate frequent rules, go into find_rules.py and insert the location of your development set and 
ASAG model in lines 477 and 479.
Additionally, the counter fitted vectors of TextFooler should be precalculated and the location
of the counterfitted-fitted-vectors.txt needs to be inserted in line 494, 
as well as the cos_sim_counter_fitting.npy file in line 502.
You can also specify where to save the frequent rules in line 631.

These frequent rules can then be filtered with the provided filter_rules.py script.
Insert the location of your frequent rules in line 14.
For manual review, set a debug point at line 50 and insert the indices of good rules into line 50.

The final rules can be used to attack an ASAG model and a specified data set.
The apply_rules_....py provide scripts, depending on the ASAG model and type of rules (baseline or else).
The main function provides examples of how the code can run.
The script prints out the total number of adversarial examples as well as the total number of afflicted 
data instances.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

In this project, the following code was used and modified:
* https://github.com/marcotcr/sears
* https://github.com/OpenNMT/OpenNMT-py
* https://github.com/jind11/TextFooler

