# DeepWeeds_ModelTuner
Code for creating, training, and fine tuning existing Keras Models on the DeepWeeds Dataset.

Download python file (.py) and open in IDE.
See line 452 to set which action to run (Build new pre-trained base, Continue to train new head, continue to fine tune base).
If action (line 452) is set to 1, see line 463 to select base model architecture.
If action (line 452) is set to 2 or 3, see line 467 to add file path to existing model file.

Initial learning rates for transfer learning and fine tuning can be found on lines 470 and 472.
