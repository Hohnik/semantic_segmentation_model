# Tasks

## Dataset

Downloading the 'fine' anotations only is NOT enougth -> leftImg8bit + gtFine

Cityscapes class tells you this:
    parameter root: Root directory of dataset where directory `leftImg8bit` and `gtFine` or `gtCoarse` are located.  


## Data Pipeline

[x] Resize
[x] Normalize
[x] Class indices -> found them on github
[x] Validation

## Modeling

!!! Your final report (see below) should include a table listing the spatial resolution, channel count, and number of blocks at each stage.
[ ] Dropout
[ ] Batch Norm
