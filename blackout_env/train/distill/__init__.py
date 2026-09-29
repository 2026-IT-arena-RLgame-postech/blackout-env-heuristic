"""Policy distillation of a relabelling teacher (heuristics/teach.py, v17_teach.py) into MyModel.

A separate pipeline from QMIX pretraining (offline_pretrain.py / qmix_trainer.py): no reward, no
Q-learning, only DAgger -- the learner plays, the teacher labels every state the learner's team
reaches, and the net is fitted to the aggregate of all labels so far.

  dataset.py  shard format (one .npz per match) and the in-memory aggregate the trainer samples
  collect.py  labelled matches, and the Unity worker pool that plays them
  loss.py     set-valued cross-entropy over the net's 8 direction logits
  run.py      the asynchronous DAgger loop (``python -m blackout_env.train.distill.run``)
"""
