# Symptom Extraction Demo

## Dataset description

The datasets required for the demo are available in the `data` folder. The dataset is based on simulated medical conversations obtained from [A dataset of simulated patient-physician medical interviews with a focus on respiratory cases, Fareez, F. _et al._ (2022)](https://www.nature.com/articles/s41597-022-01423-1)

The dataset is a .csv file with at least two columns:
- `text`: the transcript of the patient-doctor interaction.
- `label`: the expected output. For symptom tracking (binary case), the label should be 'Positive' if any symptom is mentioned in the text, and 'Negative' otherwise. For symptom extraction (multi-label case), the label should be a semicolon-separated string of symptoms, e.g. `fever;other;trouble drinking fluids`

The dataset can include any additional metadata as columns, in this case you'll find the `source` field, which corresponds to the ID of the transcript from which the text segment was taken from.

## License 
This repository is made available under the terms of the GNU General Public License version 2. You are free to use, modify, and distribute the code under these terms.

For those interested in using this project under a different licensing arrangement, commercial license options are also available. Please contact charlotta_lindvall@dfci.harvard.edu for more information.

Copyright © 2024 Dana-Farber Cancer Institute, Inc. All Rights Reserved.
