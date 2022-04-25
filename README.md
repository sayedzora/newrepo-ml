# Steps to train and genarate the weights
1. Model is trained by the labelled data in `Phase 1`.
2. These trained model generates prediction for the unlabelled data.
3. The labelled and unlabelled data both are used together for the `Phase 2` training.
4. The final weight is uploaded to the Google Drive.

# Steps for Evaluation
1. Extract the zip file.
2. Download the weight from [here](https://drive.google.com/drive/folders/1i4qCqvSRt-TNNSwHjULB3SgS2kVrJTMP?usp=sharing) (and preferably place it in root folder)
3. As the *Path of the pretrained weight* can vary from computer to computer,  we have decided to keep that argument of the main function in `evaluate_script.py` file. . By default the weight  is assumed to be placed in the root folder. Change this according to your need 

```
if __name__ == "__main__":
    main(load = "./best_weight.pth")
```

4. Run `evaluate_script.py` file. 

