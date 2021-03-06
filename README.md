# EvalML <3 Rapids in WiDS 2021
## How to Use
Simply clone the repository `git clone https://github.com/RazHoshia/Wids2021EvalMLRapids.git` and run the code with the custom Rapids' implementation (check out the [example file](https://github.com/RazHoshia/Wids2021EvalMLRapids/blob/main/example.py)).<br>
<b>Important!</b> we currently support only classification problems.
## Combining EvalML and Rapids during the WiDS2021 competition
Our team participated in the WiDS2021 challenge (5th place) and our solution combines a variety of AutoML tools (check out our paper). When we saw that EvalML gives the best results, we wanted to try and expand its search space. Following the competition, we figured out that the powerful combination of EvalML with Rapids can expand the search space  while keeping relativly short time frames.<br>
This code is a simple but working implementation of the basic sklearn estimators that gives us the best results as Rapids models.
While it's not a production-ready implementation, we hope it will help promote the research on the subject.<br>
We release the code under the Apache License 2.0, hoping it will help develop new Rapids estimators and for EvalML and other sklearn based tools.<br>
If you have any questions, feel free to contact us.
## Refrences
- https://evalml.alteryx.com/en/stable/user_guide/components.html
- https://evalml.alteryx.com/en/stable/user_guide/pipelines.html
- https://scikit-learn.org/stable/developers/develop.html
- https://docs.rapids.ai/api/cuml/stable/api.html#regression-and-classification
- https://www.kaggle.com/c/widsdatathon2021/
