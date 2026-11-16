# Data license

`alerts_test.jsonl`, `alerts_dev.jsonl` and `alerts_reference.jsonl` contain transaction rows from
**PaySim** and are therefore shared under the
[Creative Commons Attribution-ShareAlike 4.0 International license (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

**Source:** E. A. Lopez-Rojas, A. Elmir and S. Axelsson, "PaySim: A financial mobile money simulator
for fraud detection", 28th European Modeling and Simulation Symposium (EMSS), Larnaca, Cyprus, 2016.
Dataset: <https://www.kaggle.com/datasets/ealaxi/paysim1>.

**Changes made:** rows were sampled into pools (see `PAYSIM_POOLS` in `generate_dataset.py`), the
`isFraud` and `isFlaggedFraud` columns were removed from the model input, and each row was combined
with generated session, device, identity and velocity signals and a scenario label.
