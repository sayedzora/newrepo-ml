# Training Steps and Details

1. Run `training_script.py`. It will generate initial training weight from only labelled data.Get latest weight from it.
2. Run `evaluation_script.py` with latest weight path if you want to submit only this much work. 
3. To work further, run `ssl_annotations.py` to generate annotations for unlabeled data. These .yml files will be generated in a certain folder in root folder.
4. Now after loading the latest weight run `training_script_2.py` to train together with both labelled and unlabelled data. New weight will be generated. 
5. Run `evaluation_script.py` with this final weight and update result on validation data in the readme file.


# Evaluation Steps and Details

1. Extract the zip file.
2. Download the weight from [here](https://drive.google.com/file/d/10BEbYILIA3a-TGrZ8bpURDl1nst1MqLa/view?usp=sharing) (and preferably place it in root folder)
3. As the *Path of the pretrained weight* can vary,  we have decided to keep that argument of the main function in `evaluate_script.py` file. . By default the weight  is assumed to be placed in the root folder. Here assuming the name `best_weight.pth`, Change this path as necessary 

```
if __name__ == "__main__":
    main(load = "./best_weight.pth")
```

4. Run `evaluate_script.py` file. 

# Evaluation Results

IoU metric: bbox
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.230
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.225
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.053
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.116
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.263
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.349
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.447
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.098
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.502
```
